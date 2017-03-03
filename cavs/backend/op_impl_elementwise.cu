#include "cavs/backend/op_impl_elementwise.cuh"
#include "cavs/backend/op_impl_elementwise_common.h"
#include "cavs/backend/functors_elementwise.h"

namespace backend {

REGISTER_OP_IMPL_BUILDER(Key("Abs").Device("GPU"),
    CudaUnaryOpInstance(math::Abs, float));
REGISTER_OP_IMPL_BUILDER(Key("Neg").Device("GPU"),
    CudaUnaryOpInstance(math::Neg, float));
REGISTER_OP_IMPL_BUILDER(Key("Add").Device("GPU"),
    CudaBinaryOpInstance(math::Add, float));
REGISTER_OP_IMPL_BUILDER(Key("Sub").Device("GPU"),
    CudaBinaryOpInstance(math::Sub, float));
REGISTER_OP_IMPL_BUILDER(Key("Mul").Device("GPU"),
    CudaBinaryOpInstance(math::Mul, float));
REGISTER_OP_IMPL_BUILDER(Key("Scal").Device("GPU"),
    CudaBinaryScalarOpInstance(math::Mul, float));
REGISTER_OP_IMPL_BUILDER(Key("Square").Device("GPU"),
    CudaUnaryOpInstance(math::Square, float));
REGISTER_OP_IMPL_BUILDER(Key("Fill").Device("GPU"),
    CudaUnaryScalarOpInstance(math::Assign, float));

} //namespace backend
