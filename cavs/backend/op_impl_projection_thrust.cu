#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace backend {

using ::midend::Allocator;
using ::midend::Tensor;

template <typename T>
class ProjectionOpThrust: public OpImpl {
 public:
  explicit ProjectionOpThrust(const OpDef& def);
  ~ProjectionOpThrust();
  void Compute(OpContext* context) override;

 private:
  /*Allocator* alloc_;*/
  T* workspace;
};

template <typename T>
ProjectionOpThrust<T>::ProjectionOpThrust(const OpDef& def)
    : OpImpl(def), workspace(NULL) {
  /*alloc_ = GetAllocator(DeviceTypeToString(GPU));*/
}

template <typename T>
ProjectionOpThrust<T>::~ProjectionOpThrust() {
  /*if (workspace)*/
    /*alloc_->Deallocate<char>((char*)workspace); */
}

template <typename T>
__global__ void FindMax(T *out, T* mu, T* mu_scan, int n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if ((mu[i] + (1.f - mu_scan[i])/(i+1)) > 0) {
      if (i == n-1 || mu[i+1] + (1.f - mu_scan[i+1]/(i+2) < 0)) 
        *out = 1.f/i*(1-mu_scan[i]);
    }
  }
}

template <typename T>
__global__ void GetOutput(T *x, T* y, T lamda, int n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T tmp = y[i] + lamda;
    if (tmp > 0)
      x[i] = tmp;
    else
      x[i] = 0;
  }
}

template <typename T>
void ProjectionOpThrust<T>::Compute(OpContext* context) {
  const Tensor& var_in = context->Input(0);
  Tensor* var_out = context->Output(0);
  
  /*if (!workspace)*/
    /*workspace = alloc_->Allocate<char>(var_in.count()*sizeof(T));*/
  /*checkCudaError(cudaMemcpy(workspace,*/
        /*var_in.data<T>(), var_in.count()*sizeof(T),*/
        /*cudaMemcpyDeviceToDevice));*/
  int n = var_in.count();
  thrust::device_ptr<T> dev_ptr(var_in.data<T>());
  thrust::device_vector<T> mu(dev_ptr, dev_ptr+n);
  thrust::sort(mu, mu+n);
  thrust::device_vector<T> mu_scan(n);
  thrust::inclusive_scan(mu, mu+n, mu_scan);
  T lamda;
  FindMax<T><<<THREADS_PER_BLOCK, BLOCKS_PER_GRID(n)>>>(&lamda,
      thrust::raw_pointer_cast(mu.data()), 
      thrust::raw_pointer_cast(mu_scan.data()),
      n);
  GetOutput<T><<<THREADS_PER_BLOCK, BLOCKS_PER_GRID(n)>>>(
      var_out->mutable_data<T>(),
      var_in.data<T>(), 
      lamda,
      n);
  checkCudaError(cudaDeviceSynchronize());
}

} //namespace backend
