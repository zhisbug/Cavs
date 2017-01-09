#include "cavs/backend/op_impl_placeholder.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

template <typename T> 
struct CUDAMemCopy {
  static void Compute(T* out, const T* in, size_t n) {
    checkCudaError(cudaMemcpy(out, in, n*sizeof(T), cudaMemcpyHostToDevice));
  }
};

REGISTER_OP_IMPL_BUILDER(Key("Placeholder").Device("GPU"), PlaceholderOpImpl<CUDAMemCopy<float>, float>);

} //namespace backend

