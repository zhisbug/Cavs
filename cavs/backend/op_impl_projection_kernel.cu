#include "cavs/backend/op_impl.h"
#include "cavs/backend/functor_sort_scan.cuh"
#include "cavs/midend/allocator.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

using ::midend::Allocator;
using ::midend::GetAllocator;
using ::midend::DeviceTypeToString;
using ::midend::Tensor;

template <typename T>
class ProjectionOpKernel: public OpImpl {
 public:
  explicit ProjectionOpKernel(const OpDef& def);
  ~ProjectionOpKernel();
  void Compute(OpContext* context) override;

 private:
  Allocator* alloc_;
  T* workspace_sort;
  T* workspace_scan;
  T* lamda;
};

template <typename T>
ProjectionOpKernel<T>::ProjectionOpKernel(const OpDef& def)
    : OpImpl(def), workspace_sort(NULL), workspace_scan(NULL), lamda(NULL) {
  alloc_ = GetAllocator(DeviceTypeToString(GPU));
}

template <typename T>
ProjectionOpKernel<T>::~ProjectionOpKernel() {
  if (lamda)
    alloc_->Deallocate<char>((char*)lamda); 
  if (workspace_sort)
    alloc_->Deallocate<char>((char*)workspace_sort); 
  if (workspace_scan)
    alloc_->Deallocate<char>((char*)workspace_scan); 
}

template <typename T>
__global__ void BatchedFindMax(T *out, const T* mu, const T* mu_scan, int N) {
  int offset = blockIdx.x*N;
  for (int round = 0; round < (N+blockDim.x-1)/blockDim.x; round++) {
    int offset_within_vec = threadIdx.x + round*blockDim.x;
    int idx = offset + offset_within_vec;
    if (offset_within_vec < N) {  
      if ((mu[idx] + (1.f - mu_scan[idx])/(idx+1)) > 0) {
        if (idx == N-1 || mu[idx+1] + (1.f - mu_scan[idx+1]/(idx+2) < 0)) {
          out[blockIdx.x] = (1-mu_scan[idx])/(idx+1);
        }
      }
    }
  }
}

template <typename T>
__global__ void BatchedGetOutput(T *x, const T* y, const T* lamda, int N) {
  int offset = blockIdx.x*N;
  for (int round = 0; round < (N+blockDim.x-1)/blockDim.x; round++) {
    int offset_within_vec = threadIdx.x + round*blockDim.x;
    int idx = offset + offset_within_vec;
    if (offset_within_vec < N) {  
      T tmp = y[idx] + lamda[blockIdx.x];
      if (tmp > 0)
        x[idx] = tmp;
      else
        x[idx] = 0;
    }
  }
}

template <typename T>
void ProjectionOpKernel<T>::Compute(OpContext* context) {
  const Tensor& var_in = context->Input(0);
  Tensor* var_out = context->Output(0);
  CHECK(var_in.dims(0) == var_out->dims(0));
  CHECK(var_in.count() == var_out->count());

  int batch = var_in.dims(0);
  int N = var_in.count()/var_in.dims(0);
  if (!lamda)
    lamda = alloc_->Allocate<T>(var_in.dims(0));
  if (!workspace_sort)
    workspace_sort = alloc_->Allocate<T>(var_in.count());
  if (!workspace_scan)
    workspace_scan = alloc_->Allocate<T>(var_in.count());

  {
    BatchedOddEvenSort(
        workspace_sort, var_in.data<T>(), false, N, batch);
  }

  {
    const int MAX_THREADS_IN_BLOCK = 1 << 10;
    int threadsPerBlock = (MAX_THREADS_IN_BLOCK > N) ? N : MAX_THREADS_IN_BLOCK;
    int blocksPerGrid = batch;
    BatchedScan(workspace_scan, workspace_sort, N, batch);
    BatchedFindMax<T><<<blocksPerGrid, threadsPerBlock>>>(
        lamda, workspace_sort, workspace_scan, N);
    BatchedGetOutput<T><<<blocksPerGrid, threadsPerBlock>>>(
        var_out->mutable_data<T>(),
        var_in.data<T>(),
        lamda, N);
  }

  /*var_out->DebugNumerical<T>();*/
  /*checkCudaError(cudaDeviceSynchronize());*/
}

REGISTER_OP_IMPL_BUILDER(Key("Simplex").Device("GPU"), ProjectionOpKernel<float>);

} //namespace backend
