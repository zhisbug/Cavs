#include "cavs/backend/op_impl_variable.h"
#include "cavs/backend/op_impl_elementwise.cuh"
#include "cavs/backend/functors_elementwise.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU"),
    VariableOpImpl<CUDAUnaryConstScalarFunctor<math::Assign<float>, float>, float>);

} //namespace backend
