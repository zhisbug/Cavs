#include "cavs/backend/op_impl_elementwise.cuh"
#include "cavs/backend/op_impl_elementwise_common.h"
#include "cavs/backend/functor_elementwise.h"

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
REGISTER_OP_IMPL_BUILDER(Key("Div").Device("GPU"),
    CudaBinaryOpInstance(math::Div, float));
REGISTER_OP_IMPL_BUILDER(Key("Square").Device("GPU"),
    CudaUnaryOpInstance(math::Square, float));
REGISTER_OP_IMPL_BUILDER(Key("Scal").Device("GPU"),
    CudaBinaryOpInstance(math::Mul, float));
/*REGISTER_OP_IMPL_BUILDER(Key("Fill").Device("GPU"),*/
    /*CudaUnaryOpInstance(math::Assign, float));*/

REGISTER_OP_IMPL_BUILDER(Key("Equal").Device("GPU"),
    CudaBinaryOpInstance(math::Equal, float));

//For partial-add, we have reset the augend tensor to 0 in each iteration
REGISTER_OP_IMPL_BUILDER(Key("PartialAdd").Device("GPU"),
    CudaPartialAddBinaryOpInstance(math::Add, float));

} //namespace backend
