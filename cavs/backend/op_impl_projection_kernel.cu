#include "cavs/backend/op_impl.h"
#include "cavs/backend/op_impl_projection.cuh"
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
__global__ void BatchedFindMax(T *out, const T* mu, const T* mu_scan, int n) {
  int idx = threadIdx.x + blockIdx.x*n;
  if (threadIdx.x < n) {  
    if ((mu[idx] + (1.f - mu_scan[idx])/(idx+1)) > 0) {
      if (idx == n-1 || mu[idx+1] + (1.f - mu_scan[idx+1]/(idx+2) < 0)) 
        out[blockIdx.x] = (1-mu_scan[idx])/(idx+1);
    }
  }
}

template <typename T>
__global__ void BatchedGetOutput(T *x, const T* y, const T* lamda, int n) {
  int idx = threadIdx.x + blockIdx.x*n;
  if (threadIdx.x < n) {  
    T tmp = y[idx] + lamda[blockIdx.x];
    if (tmp > 0)
      x[idx] = tmp;
    else
      x[idx] = 0;
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
    const int SHARE_SIZE_LIMIT = 1 << 13;
    int threadsPerBlock = 1;
    while (threadsPerBlock < N) { threadsPerBlock <<= 1; }
    threadsPerBlock >>= 1;
    int blocksPerGrid = batch;
    BatchedMergeSort<T, SHARE_SIZE_LIMIT><<<blocksPerGrid, threadsPerBlock>>>(
        workspace_sort, var_in.data<T>(), N, false);
  }

  {
    const int SHARE_SIZE_LIMIT = 1 << 13;
    int threadsPerBlock = N;
    int blocksPerGrid = batch;
    BatchedScan<SHARE_SIZE_LIMIT><<<blocksPerGrid, threadsPerBlock>>>(
        workspace_scan, workspace_sort, N);
    BatchedFindMax<T><<<blocksPerGrid, threadsPerBlock>>>(
        lamda, workspace_sort, workspace_scan, N);
    BatchedGetOutput<T><<<blocksPerGrid, threadsPerBlock>>>(
        var_out->mutable_data<T>(),
        var_in.data<T>(),
        lamda, N);
  }
  for (int i = 0; i < vec_num; i++) {
    thrust::device_ptr<T> dev_ptr(const_cast<T*>(var_in.data<T>()+i*vec_size));
    thrust::device_vector<T> mu(dev_ptr, dev_ptr+vec_size);
    thrust::sort(mu.begin(), mu.end());
    thrust::device_vector<T> mu_scan(vec_size);
    thrust::inclusive_scan(mu.begin(), mu.end(), mu_scan.begin());
    FindMax<T><<<BLOCKS_PER_GRID(vec_size), THREADS_PER_BLOCK>>>(lamda,
        thrust::raw_pointer_cast(mu.data()), 
        thrust::raw_pointer_cast(mu_scan.data()),
        vec_size);
    GetOutput<T><<<BLOCKS_PER_GRID(vec_size), THREADS_PER_BLOCK>>>(
        var_out->mutable_data<T>()+i*vec_size,
        var_in.data<T>()+i*vec_size,
        lamda, vec_size);
  }

  /*var_out->DebugNumerical<T>();*/
  /*checkCudaError(cudaDeviceSynchronize());*/
}


} //namespace backend
