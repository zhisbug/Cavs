#include "cavs/backend/op_impl.h"
#include "cavs/midend/allocator.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thread>

using std::vector;
using std::thread;

namespace backend {

using ::midend::Allocator;
using ::midend::GetAllocator;
using ::midend::DeviceTypeToString;
using ::midend::Tensor;

template <typename T>
class ProjectionOpThrust: public OpImpl {
 public:
  explicit ProjectionOpThrust(const OpDef& def);
  ~ProjectionOpThrust();
  void Compute(OpContext* context) override;

 private:
  Allocator* alloc_;
  T* lamda;
  /*const int THREAD_POOL_SIZE;*/
  /*void thread_func(T* out, const T* in, int begin, int stride);*/
};

template <typename T>
ProjectionOpThrust<T>::ProjectionOpThrust(const OpDef& def)
    : OpImpl(def), lamda(NULL)/*, THREAD_POOL_SIZE(32)*/ {
  alloc_ = GetAllocator(DeviceTypeToString(GPU));
  if (!lamda)
    lamda = alloc_->Allocate<T>(1);
}

template <typename T>
ProjectionOpThrust<T>::~ProjectionOpThrust() {
  if (lamda)
    alloc_->Deallocate<char>((char*)lamda); 
}

template <typename T>
__global__ void FindMax(T *out, const T* mu, const T* mu_scan, int n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if ((mu[i] + (1.f - mu_scan[i])/(i+1)) > 0) {
      if (i == n-1 || mu[i+1] + (1.f - mu_scan[i+1]/(i+2) < 0)) 
        *out = (1-mu_scan[i])/(i+1);
    }
  }
}

template <typename T>
__global__ void GetOutput(T *x, const T* y, const T* lamda, int n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T tmp = y[i] + *lamda;
    if (tmp > 0)
      x[i] = tmp;
    else
      x[i] = 0;
  }
}

template <typename T>
void thread_func(T* out, const T* in, int N, int offset, int stride,
                 T* lamda) {
  for (int i = offset; i < stride; i++) {
    thrust::device_ptr<T> dev_ptr(const_cast<T*>(in+i*N));
    thrust::device_vector<T> mu(dev_ptr, dev_ptr+N);
    thrust::sort(mu.begin(), mu.end(), thrust::greater<T>());
    thrust::device_vector<T> mu_scan(N);
    thrust::inclusive_scan(mu.begin(), mu.end(), mu_scan.begin());
    FindMax<T><<<BLOCKS_PER_GRID(N), THREADS_PER_BLOCK>>>(lamda,
        thrust::raw_pointer_cast(mu.data()), 
        thrust::raw_pointer_cast(mu_scan.data()),
        N);
    GetOutput<T><<<BLOCKS_PER_GRID(N), THREADS_PER_BLOCK>>>(
        out+i*N,
        in+i*N, 
        lamda,
        N);
  }

}

template <typename T>
void ProjectionOpThrust<T>::Compute(OpContext* context) {
  const Tensor& var_in = context->Input(0);
  Tensor* var_out = context->Output(0);
  CHECK(var_in.dims(0) == var_out->dims(0));
  CHECK(var_in.count() == var_out->count());

  int vec_num = var_in.dims(0);
  int vec_size = var_in.count()/var_in.dims(0);
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

/*REGISTER_OP_IMPL_BUILDER(Key("Simplex").Device("GPU"), ProjectionOpThrust<float>);*/

} //namespace backend
