#include "cavs/backend/placeholder_op.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

template <typename T> 
struct CUDAMemCopy {
  static void Compute(T* out, const T* in, size_t n) {
    checkCudaError(cudaMemcpy(out, in, n*sizeof(T), cudaMemcpyHostToDevice));
  }
};

REGISTER_OP_BUILDER(Key("Placeholder").Device("GPU"), PlaceholderOp<CUDAMemCopy<float>, float>);

} //namespace backend

