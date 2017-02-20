#include "cavs/backend/op_impl_elementwise_common.h"
#include "cavs/backend/functors_elementwise.h"
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
    UnaryKernel<OP, T><<<THREADS_PER_BLOCK, BLOCKS_PER_GRID(n)>>>(
        out, inp, n);
  }
};

template <typename OP, typename T>
struct CUDABinaryFunctor {
  static void Compute(T* out, const T* inp0, const T* inp1, size_t n) {
    BinaryKernel<OP, T><<<THREADS_PER_BLOCK, BLOCKS_PER_GRID(n)>>>(
        out, inp0, inp1, n);
  }
};

template <typename OP, typename T>
struct CUDABinaryScalarFunctor {
  static void Compute(T* out, const T* inp0, const T* inp1, size_t n) {
    BinaryKernel<OP, T><<<THREADS_PER_BLOCK, BLOCKS_PER_GRID(n)>>>(
        out, inp0, inp1, n);
  }
};

#define CudaUnaryOpInstance(math, dtype)    \
    UnaryOp<CUDAUnaryFunctor<math<dtype>, dtype>, dtype> 
#define CudaBinaryOpInstance(math, dtype)   \
    BinaryOp<CUDABinaryFunctor<math<dtype>, dtype>, dtype> 
#define CudaBinaryScalarOpInstance(math, dtype)   \
    BinaryOp<CUDABinaryScalarFunctor<math<dtype>, dtype>, dtype> 

REGISTER_OP_IMPL_BUILDER(Key("Abs").Device("GPU"),
    CudaUnaryOpInstance(math::Abs, float));
REGISTER_OP_IMPL_BUILDER(Key("Neg").Device("GPU"),
    CudaUnaryOpInstance(math::Neg, float));
REGISTER_OP_IMPL_BUILDER(Key("Square").Device("GPU"),
    CudaUnaryOpInstance(math::Square, float));
REGISTER_OP_IMPL_BUILDER(Key("Add").Device("GPU"),
    CudaBinaryOpInstance(math::Add, float));
REGISTER_OP_IMPL_BUILDER(Key("Sub").Device("GPU"),
    CudaBinaryOpInstance(math::Sub, float));
REGISTER_OP_IMPL_BUILDER(Key("Mul").Device("GPU"),
    CudaBinaryOpInstance(math::Mul, float));
REGISTER_OP_IMPL_BUILDER(Key("Scal").Device("GPU"),
    CudaBinaryScalarOpInstance(math::Mul, float));

} //namespace backend
