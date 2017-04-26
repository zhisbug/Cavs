#include "cavs/backend/op_impl_mpi_functor.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/midend/allocator.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/mpi_types.h"
#include "cavs/util/op_util.h"

namespace backend {

using ::midend::Tensor;

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
    cpu_buffer.Rebase(::midend::GetAllocator(::midend::DeviceTypeToString(CPU)), inp);
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

template<typename T>
void MPIBcastOpImpl<T>::Compute(OpContext* context) {
  Tensor* out = context->Output(0);
  if (out->device_type() != CPU) {
    Tensor cpu_buffer;
    cpu_buffer.Rebase(::midend::GetAllocator(::midend::DeviceTypeToString(CPU)), *out);
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

REGISTER_OP_IMPL_BUILDER(Key("MPIAllReduce").Device("CPU"), MPIAllReduceOpImpl<float>);
REGISTER_OP_IMPL_BUILDER(Key("MPIBcast").Device("CPU"), MPIBcastOpImpl<float>);

} //namespace backend
