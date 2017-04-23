#ifndef CAVS_BACKEND_FUNCTOR_FILLER_CUH_
#define CAVS_BACKEND_FUNCTOR_FILLER_CUH_

#include "cavs/backend/functor_filler.h"
#include "cavs/backend/op_impl_elementwise.cuh"
#include "cavs/backend/cuda_common.h"
#include "cavs/util/op_util.h"

#include <vector>

namespace backend {

template <typename OP, typename T>
struct CudaFiller : Filler<OP, T> {
  CudaFiller(const OpDef& op_def) : Filler<OP, T>(op_def) {}
  FORCE_INLINE void Compute(T* buf, int N) override {
    std::vector<T> cpu_buf(N);
    Filler<OP, T>::Compute(cpu_buf.data(), N);
    /*for (int i = 0; i < N; i += this->stride_) {*/
      /*OP::Compute(cpu_buf.data()+i, (i+this->stride_>N) ? (N-i) : this->stride_);*/
    /*}*/
    checkCudaError(cudaMemcpy(buf, cpu_buf.data(), N*sizeof(T),
                              cudaMemcpyHostToDevice));
  }
};

template <typename OP, typename T>
struct CudaConstantFiller {
  CudaConstantFiller(const OpDef& op_def) {
    value = GetSingleArg<T>(op_def, "const_value");
  }
  FORCE_INLINE void Compute(T* buf, size_t n) {
    UnaryConstScalarKernel<OP, T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        buf, value, n);
  }
  T value;
};

} //namspace backend

#endif
