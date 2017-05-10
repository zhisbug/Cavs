#ifndef CAVS_BACKEND_FUNCTOR_FILLER_CUH_
#define CAVS_BACKEND_FUNCTOR_FILLER_CUH_

#include "cavs/backend/functor_filler.h"
#include "cavs/backend/op_impl_elementwise.cuh"
#include "cavs/backend/cuda_common.h"
#include "cavs/util/op_util.h"

#include <vector>

namespace backend {

template <typename FILLER, typename T>
struct CudaFiller {
  CudaFiller(const OpDef& op_def) : filler_(op_def) {}
  FORCE_INLINE void Compute(T* buf, int N) override {
    std::vector<T> cpu_buf(N);
    filler_.Compute(cpu_buf.data(), N);
    checkCudaError(cudaMemcpy(buf, cpu_buf.data(), N*sizeof(T),
                              cudaMemcpyHostToDevice));
  }
 private:
  FILLER filler_;
};

/*template <typename OP, typename T>*/
/*struct CudaConstantFiller {*/
  /*CudaConstantFiller(const OpDef& op_def) {*/
    /*value = GetSingleArg<T>(op_def, "const_value");*/
  /*}*/
  /*FORCE_INLINE void Compute(T* buf, size_t n) {*/
    /*UnaryConstScalarKernel<OP, T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(*/
        /*buf, value, n);*/
  /*}*/
  /*T value;*/
/*};*/

} //namspace backend

#endif
