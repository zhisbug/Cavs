#include "cavs/backend/op_impl_variable.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

template <typename T> 
struct CUDAZeroFiller {
  static void Compute(T* out, size_t n) {
    checkCudaError(cudaMemset(out, 0, n*sizeof(T)));
  }
};

REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU"), VariableOpImpl<CUDAZeroFiller<float>, float>);

} //namespace backend
