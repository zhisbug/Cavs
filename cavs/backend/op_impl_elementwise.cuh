#ifndef CAVS_BACKEND_OP_IMPL_ELEMENTWISE_CUH_
#define CAVS_BACKEND_OP_IMPL_ELEMENTWISE_CUH_

#include "cavs/backend/cuda_common.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

template <typename OP, typename T, typename U> 
__global__ void UnaryKernel(T* out, const U* inp, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(inp[i]); 
  } 
}

template <typename OP, typename T, typename U> 
__global__ void UnaryScalarKernel(T* out, const U* value, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(*value); 
  } 
}

template <typename OP, typename T, typename U> 
__global__ void UnaryConstScalarKernel(T* out, const U value, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(value); 
  } 
}

template <typename OP, typename T, typename U=T>
struct CUDAUnaryFunctor {
  static void Compute(T* out, size_t n_out, const U* inp, size_t n_inp) {
    if (n_inp == 1) {
      UnaryScalarKernel<OP, T, U><<<BLOCKS_PER_GRID(n_out), THREADS_PER_BLOCK>>>(
          out, inp, n_out);
    }else {
      UnaryKernel<OP, T, U><<<BLOCKS_PER_GRID(n_out), THREADS_PER_BLOCK>>>(
          out, inp, n_out);
    }
    checkCudaError(cudaGetLastError());
  }
};

template <typename OP, typename T, typename U=T> 
struct CUDAUnaryConstScalarFunctor {
  static void Compute(T* out, const U value, size_t n) {
    UnaryConstScalarKernel<OP, T, U><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        out, value, n);
    checkCudaError(cudaGetLastError());
  }
};

template <typename OP, typename T, typename U> 
__global__ void BinaryKernel(T* out, const U* inp0, const U* inp1, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(inp0[i], inp1[i]); 
  } 
}

template <typename OP, typename T, typename U> 
__global__ void BinaryScalarKernel(T* out, const U* inp0, const U *inp1, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(inp0[i], *inp1); 
  } 
}

template <typename OP, typename T, typename U> 
__global__ void BinaryConstScalarKernel(T* out, const U* inp0, const U inp1, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = OP::Compute(inp0[i], inp1); 
  } 
}

template <typename OP, typename T, typename U=T>
struct CUDABinaryFunctor {
  static void Compute(T* out, size_t n_out,
      const U* inp0, size_t n_inp0, const U* inp1, size_t n_inp1) {
    if (n_out == n_inp0 && n_inp0 == n_inp1) {
      BinaryKernel<OP, T, U><<<BLOCKS_PER_GRID(n_out), THREADS_PER_BLOCK>>>(
          out, inp0, inp1, n_out);
    }else if (n_inp1 == 1 && n_out == n_inp0) {
      BinaryScalarKernel<OP, T, U><<<BLOCKS_PER_GRID(n_out), THREADS_PER_BLOCK>>>(
          out, inp0, inp1, n_out);
    }else if (n_inp0 == 1 && n_out == n_inp1) {
      BinaryScalarKernel<OP, T, U><<<BLOCKS_PER_GRID(n_out), THREADS_PER_BLOCK>>>(
          out, inp1, inp0, n_out);
    }else {
      LOG(FATAL) << "Unrecognized Pattern";
    }
    checkCudaError(cudaGetLastError());
  }
};

template <typename OP, typename T, typename U=T>
struct CUDABinaryConstScalarFunctor {
  static void Compute(T* out, size_t n_out, const U* inp0, size_t n_inp0,
      const U inp1) {
    CHECK(n_out == n_inp0);
    BinaryConstScalarKernel<OP, T, U><<<BLOCKS_PER_GRID(n_out), THREADS_PER_BLOCK>>>(
        out, inp0, inp1, n_out);
    checkCudaError(cudaGetLastError());
  }
};

#define CudaUnaryOpInstance(math, dtype)    \
    UnaryOp<CUDAUnaryFunctor<math<dtype>, dtype>, dtype>
#define CudaUnaryConstScalarOpInstance(math, dtype)    \
    UnaryOp<CUDAUnaryConstScalarFunctor<math<dtype>, dtype>, dtype>
#define CudaBinaryOpInstance(math, dtype)   \
    BinaryOp<CUDABinaryFunctor<math<dtype>, dtype>, dtype>
/*#define CudaBinaryScalarOpInstance(math, dtype)   \*/
    /*BinaryOp<CUDABinaryScalarFunctor<math<dtype>, dtype>, dtype> */

} //namespace backend

#endif
