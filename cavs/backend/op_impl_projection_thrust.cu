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
  T* workspace;
  T* lamda;
  const int THREAD_POOL_SIZE;
  /*void thread_func(T* out, const T* in, int begin, int stride);*/
};

template <typename T>
ProjectionOpThrust<T>::ProjectionOpThrust(const OpDef& def)
    : OpImpl(def), workspace(NULL), lamda(NULL), THREAD_POOL_SIZE(32) {
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
void thread_func(T* out, const T* in, int N, int offset, int stride) {
  for (int i = offset; i < stride; i++) {
    thrust::device_ptr<T> dev_ptr(const_cast<T*>(in+i*N));
    thrust::device_vector<T> mu(dev_ptr, dev_ptr+N);
    thrust::sort(mu.begin(), mu.end(), thrust::greater<T>());
    if (i == 0) {
      std::vector<T> h_vec(10); 
      checkCudaError(cudaMemcpy(h_vec.data(),
            thrust::raw_pointer_cast(mu.data()), 10*sizeof(T),
            cudaMemcpyDeviceToHost));
      for (int i = 0; i < 10; i++) {
        LOG(INFO) << "sorted[" << i << "]: "
                  << h_vec[i];
      }
    }
    thrust::device_vector<T> mu_scan(n);
    thrust::inclusive_scan(mu.begin(), mu.end(), mu_scan.begin());
    /*checkCudaError(cudaDeviceSynchronize());*/
    FindMax<T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(lamda,
        thrust::raw_pointer_cast(mu.data()), 
        thrust::raw_pointer_cast(mu_scan.data()),
        n);
    /*if (i <= 10) {*/
      /*LOG(INFO) << "\n\n\n";*/
      /*T lamda_h; */
      /*checkCudaError(cudaMemcpy(&lamda_h, lamda, sizeof(T),*/
            /*cudaMemcpyDeviceToHost));*/
      /*LOG(INFO) << "lamda = " << lamda_h << "\n\n\n";*/
    /*}*/
    /*checkCudaError(cudaDeviceSynchronize());*/
    GetOutput<T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        var_out->mutable_data<T>()+i*n,
        var_in.data<T>()+i*n, 
        lamda,
        n);
  }

}

template <typename T>
void ProjectionOpThrust<T>::Compute(OpContext* context) {
  const Tensor& var_in = context->Input(0);
  Tensor* var_out = context->Output(0);
  /*if (!workspace)*/
    /*workspace = alloc_->Allocate<char>(var_in.count()*sizeof(T));*/
  /*checkCudaError(cudaMemcpy(var_out->mutable_data<T>(),*/
        /*var_in.data<T>(), var_in.count()*sizeof(T),*/
        /*cudaMemcpyDeviceToDevice));*/
  CHECK(var_in.dims(0) == var_out->dims(0));
  CHECK(var_in.count() == var_out->count());
  int N = var_in.count()/var_in.dims(0);
  vector<thread> thread_pool;
  for (int i = 0; i < var_in.dims(0)/THREAD_POOL_SIZE; i++) {
    /*LOG(INFO) << "here " << i;*/
    thread_pool.push_back(thread(&thread_func<T>,
            var_out->mutable_data<T>(), var_in.data<T>(),
            N, i*THREAD_POOL_SIZE, (i+1)*THREAD_POOL_SIZE));
  }
  for (auto& th : thread_pool)
    th.join();
  /*var_out->DebugNumerical<T>();*/
  checkCudaError(cudaDeviceSynchronize());
}

REGISTER_OP_IMPL_BUILDER(Key("Simplex").Device("GPU"), ProjectionOpThrust<float>);

} //namespace backend
