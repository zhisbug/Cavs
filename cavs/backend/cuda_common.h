#ifndef CAVS_BACKEND_CUDA_COMMON_H_
#define CAVS_BACKEND_CUDA_COMMON_H_

#include "cavs/util/macros_gpu.h"

namespace backend {

#define CUDA_1D_KERNEL_LOOP(i, n)                        \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;  \
            i < (n); i += blockDim.x * gridDim.x)

const int THREADS_PER_BLOCK = 1024;

__inline__ int BLOCKS_PER_GRID(const int N) {
    return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

class CudaCommon{
 public:
  inline static cublasHandle_t cublasHandle() { return Get()->cublasHandle_; }
  inline static cudnnHandle_t cudnnHandle() { return Get()->cudnnHandle_; }

 private:
  CudaCommon() {
    checkCublasError(cublasCreate(&cublasHandle_));
    checkCUDNNError(cudnnCreate(&cudnnHandle_));
  }
  static CudaCommon* Get() {static CudaCommon cc; return &cc;}
  cublasHandle_t cublasHandle_;
  cudnnHandle_t cudnnHandle_;
};

} //namespace backend

#endif
