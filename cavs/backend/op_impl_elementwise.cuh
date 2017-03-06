#ifndef CAVS_BACKEND_OP_IMPL_ELEMENTWISE_CUH_
#define CAVS_BACKEND_OP_IMPL_ELEMENTWISE_CUH_

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
__global__ void UnaryScalarKernel(T* out, const T* value, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(*value); 
  } 
}

template <typename OP, typename T> 
__global__ void UnaryConstScalarKernel(T* out, const T value, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(value); 
  } 
}

template <typename OP, typename T> 
struct CUDAUnaryScalarFunctor{
  static void Compute(T* out, const T* value, size_t n) {
    /*checkCudaError(cudaMemset(out, 0, n*sizeof(T)));*/
    UnaryScalarKernel<OP, T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        out, value, n);
  }
};

template <typename OP, typename T> 
struct CUDAUnaryConstScalarFunctor{
  static void Compute(T* out, const T value, size_t n) {
    /*checkCudaError(cudaMemset(out, 0, n*sizeof(T)));*/
    UnaryConstScalarKernel<OP, T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        out, value, n);
  }
};

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

#define CudaUnaryOpInstance(math, dtype)    \
    UnaryOp<CUDAUnaryFunctor<math<dtype>, dtype>, dtype>
#define CudaUnaryScalarOpInstance(math, dtype)    \
    UnaryOp<CUDAUnaryScalarFunctor<math<dtype>, dtype>, dtype>
#define CudaBinaryOpInstance(math, dtype)   \
    BinaryOp<CUDABinaryFunctor<math<dtype>, dtype>, dtype>
#define CudaBinaryScalarOpInstance(math, dtype)   \
    BinaryOp<CUDABinaryScalarFunctor<math<dtype>, dtype>, dtype> 

} //namespace backend

#endif
