#include "cavs/backend/op_impl_variable.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

template <typename T> 
__global__ void FillKernel(T* out, const T value, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = value; 
  } 
}

template <typename T> 
struct CUDAFiller {
  static void Compute(T* out, T value, size_t n) {
    /*checkCudaError(cudaMemset(out, 0, n*sizeof(T)));*/
    FillKernel<T><<<THREADS_PER_BLOCK, BLOCKS_PER_GRID(n)>>>(
        out, value, n);
  }
};

REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU"), VariableOpImpl<CUDAFiller<float>, float>);

} //namespace backend
