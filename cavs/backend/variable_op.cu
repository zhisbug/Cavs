#include "cavs/midend/macros_gpu.h"
#include "cavs/backend/variable_op.h"

namespace cavs {

template <typename T> 
struct CUDAZeroFiller {
  static void Compute(T* out, size_t n) {
    checkCudaError(cudaMemset(out, 0, n*sizeof(T)));
  }
};

REGISTER_OP_BUILDER(Key("Variable").Device("GPU"), VariableOp<CUDAZeroFiller<float>, float>);

} //namespace cavs
