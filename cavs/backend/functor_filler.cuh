#include "cavs/backend/functor_filler.h"
#include "cavs/backend/op_impl_elementwise.cuh"
#include "cavs/backend/cuda_common.h"
#include "cavs/util/op_util.h"

template <typename Op, typename T>
struct CudaFiller {
  CudaFiller(const OpDef& op_def) {}
  FORCE_INLINE void Compute(T* buf, int N) {
    vector<T> cpu_buf(N);
    for (int i = 0; i < N; i+=stride) {
      Op<T>::Compute(cpu_buf.data()+i, (i+stride>=N) ? (N-i) : stride);
    }
    checkCudaError(cudaMemcpy(buf, cpu_buf.data(), N*sizeof(T),
                              cudaMemcpyHostToDevice));
  }
};

template <typename Op, typename T>
struct CudaConstantFiller {
  CudaConstantFiller(const OpDef& op_def) {
    value = GetSingleArg<T>(op_def, "const_value");
  }
  FORCE_INLINE void Compute(T* buf, int N) {
    UnaryConstScalarKernel<OP, T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        buf, value, n);
  }
  T value;
};

