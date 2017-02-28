#include "cavs/backend/cuda_common.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

template <typename OP, typename T> 
__global__ void UnaryKernel(T* out, const T* inp, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(inp[i]); 
  } 
}

template <typename OP, typename T> 
__global__ void BinaryKernel(T* out, const T* inp0, const T* inp1, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(inp0[i], inp1[i]); 
  } 
}

template <typename OP, typename T> 
__global__ void BinaryScalarKernel(T* out, const T* inp0, const T *inp1, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(inp0[i], *inp1); 
  } 
}

template <typename OP, typename T>
struct CUDAUnaryFunctor {
  static void Compute(T* out, const T* inp, size_t n) {
    checkCudaError(cudaGetLastError());
    UnaryKernel<OP, T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        out, inp, n);
    checkCudaError(cudaGetLastError());
  }
};

template <typename OP, typename T>
struct CUDABinaryFunctor {
  static void Compute(T* out, const T* inp0, const T* inp1, size_t n) {
    checkCudaError(cudaGetLastError());
    BinaryKernel<OP, T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        out, inp0, inp1, n);
    checkCudaError(cudaGetLastError());
  }
};

template <typename OP, typename T>
struct CUDABinaryScalarFunctor {
  static void Compute(T* out, const T* inp0, const T* inp1, size_t n) {
    checkCudaError(cudaGetLastError());
    BinaryKernel<OP, T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        out, inp0, inp1, n);
    checkCudaError(cudaGetLastError());
  }
};

} //namespace backend
