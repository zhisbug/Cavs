#include "cavs/backend/op_impl_mpi_functor.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
//#include "cavs/midend/devices.h"
#include "cavs/midend/allocator.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/mpi_types.h"
#include "cavs/util/op_util.h"

namespace backend {

using ::midend::Allocator;
//using ::midend::DeviceTypeToString;
using ::midend::Tensor;
using std::vector;

//MPI is currently run on CPU and the communication is global
//CUDA-aware MPI will be supported later
//For MPI_Allreduce operator, only MPI_SUM is supported.
template <typename T>
class MPIAllReduceOpImpl: public OpImpl {
 public:
  explicit MPIAllReduceOpImpl(const OpDef& def)
    : OpImpl(def) {}
  void Compute(OpContext* context) override;
};

template<typename T>
void MPIAllReduceOpImpl<T>::Compute(OpContext* context) {
  const Tensor& inp = context->Input(0);
  Tensor* out = context->Output(0);
  //currently, we assume this
  CHECK(inp.device_type() == out->device_type());
  CHECK(inp.count() == out->count());
  if (inp.device_type() != CPU) {
    Tensor cpu_buffer; 
    cpu_buffer.Rebase(::midend::GetAllocator(DeviceTypeToString(CPU)), inp);
    cpu_buffer.SyncWith(inp);
    MPIAllReduceFunctor<T>::Compute(cpu_buffer.data<T>(), 
          cpu_buffer.mutable_data<T>(), 
          cpu_buffer.count());
    out->SyncWith(cpu_buffer);
  }else {
    MPIAllReduceFunctor<T>::Compute(inp.data<T>(), 
          out->mutable_data<T>(), 
          inp.count());
  }
}

template <typename T>
class MPIBcastOpImpl: public OpImpl {
 public:
  explicit MPIBcastOpImpl(const OpDef& def)
    : OpImpl(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
void MPIBcastOpImpl<T>::Compute(OpContext* context) {
  Tensor* out = context->Output(0);
  if (out->device_type() != CPU) {
    Tensor cpu_buffer;
    cpu_buffer.Rebase(::midend::GetAllocator(DeviceTypeToString(CPU)), *out);
    cpu_buffer.SyncWith(*out);
    //currently, we assume process0 executes the broadcast
    MPIBcastFunctor<T>::Compute(cpu_buffer.mutable_data<T>(),
          cpu_buffer.count(), 0);
    out->SyncWith(cpu_buffer);
  }else {
    MPIBcastFunctor<T>::Compute(out->mutable_data<T>(),
          out->count(), 0);
  }
}

template <typename T>
class MPISFBOpImpl: public OpImpl {
 public:
  explicit MPISFBOpImpl(const OpDef& def)
    : OpImpl(def), TransA_(false), TransB_(false),
      workspaceA(NULL), workspaceB(NULL),
      workspaceAInBytes(0), workspaceBInBytes(0) {
    for (auto& t : GetListArg<int>(op_def_, "Transpose")) {
      LOG(INFO) << "Transpose: " << t;
      if (t == 0) TransA_ = true;
      if (t == 1) TransB_ = true;
    }
    alloc_ = ::midend::GetAllocator(DeviceTypeToString(GPU));
  }
  void Compute(OpContext* context) override;

 private:
  bool TransA_;
  bool TransB_;
  Allocator* alloc_;
  void *workspaceA, *workspaceB;
  size_t workspaceAInBytes, workspaceBInBytes;
};

//C = Matmul(A, B)
template <typename T>
void MPISFBOpImpl<T>::Compute(OpContext* context) {
  const Tensor& A = context->Input(0);
  const Tensor& B = context->Input(1);
  CHECK(A.dims() == B.dims());
  CHECK(A.dims() == 2);
  int MA = (TransA_ == false)? A.dims(0) : A.dims(1);
  int KA = (TransA_ == false)? A.dims(1) : A.dims(0);
  int KB = (TransB_ == false)? B.dims(0) : B.dims(1);
  int NB = (TransB_ == false)? B.dims(1) : B.dims(0);
  CHECK(KA == KB);
  Tensor* C = context->Output(0);
  CHECK(C->dims() == 2);
  CHECK(C->dims(0) == MA);
  CHECK(C->dims(1) == NB);

  //MatMulMatCublasWrapper<T>(TransA_, TransB_,
      //MA, NB, KA, 1.f, A.data<T>(), B.data<T>(),
      //0, C->mutable_data<T>());

  int size = 0;
  checkMPIError(MPI_Comm_size(MPI_COMM_WORLD, &size));
  vector<T> recvbufA(A.count()*size);
  vector<T> recvbufB(B.count()*size);
  CHECK(A.device_type() == B.device_type());
  if (A.device_type() != CPU) {
    Tensor cpubufA;
    cpubufA.Rebase(::midend::GetAllocator(DeviceTypeToString(CPU)), A);
    cpubufA.SyncWith(A);
    MPIAllgatherFunctor<T>::Compute(cpubufA.data<T>(), A.count(),
        recvbufA.data(), A.count());
    Tensor cpubufB;
    cpubufB.Rebase(::midend::GetAllocator(DeviceTypeToString(CPU)), B);
    cpubufB.SyncWith(B);
    MPIAllgatherFunctor<T>::Compute(cpubufB.data<T>(), B.count(),
        recvbufB.data(), B.count());
  }else {
    MPIAllgatherFunctor<T>::Compute(A.data<T>(), A.count(),
        recvbufA.data(), A.count());
    MPIAllgatherFunctor<T>::Compute(B.data<T>(), B.count(),
        recvbufB.data(), B.count());
  }

  if (workspaceAInBytes < A.count()*sizeof(T)) {
    if (workspaceA)
      alloc_->Deallocate<char>((char*)workspaceA);
    workspaceA = alloc_->Allocate<char>(A.count()*sizeof(T));
    workspaceAInBytes = A.count()*sizeof(T);
  }
  if (workspaceBInBytes < B.count()*sizeof(T)) {
    if (workspaceB)
      alloc_->Deallocate<char>((char*)workspaceB);
    workspaceB = alloc_->Allocate<char>(B.count()*sizeof(T));
    workspaceBInBytes = B.count()*sizeof(T);
  }
  CHECK(workspaceA);
  CHECK(workspaceB);
  CHECK(C->device_type() == GPU);
  for (int i = 0; i < size; i++) {
    checkCudaError(cudaMemcpy(workspaceA, recvbufA.data()+A.count()*i,
          A.count()*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(workspaceB, recvbufB.data()+B.count()*i,
          B.count()*sizeof(T), cudaMemcpyHostToDevice));
    MatMulMatCublasWrapper<T>(TransA_, TransB_,
        MA, NB, KA, 1.f, (T*)workspaceA, (T*)workspaceB,
        (i == 0) ? 0.f : 1.f, C->mutable_data<T>());
  }
}

REGISTER_OP_IMPL_BUILDER(Key("MPIAllReduce").Device("GPU"), MPIAllReduceOpImpl<float>);
REGISTER_OP_IMPL_BUILDER(Key("MPIBcast").Device("GPU"),     MPIBcastOpImpl<float>);
REGISTER_OP_IMPL_BUILDER(Key("MPISFB").Device("GPU"),       MPISFBOpImpl<float>);

} //namespace backend
