#include "cavs/backend/op_impl_elementwise.cuh"
#include "cavs/backend/op_impl_elementwise_common.h"
#include "cavs/backend/functors_elementwise.h"

namespace backend {

#define CudaUnaryOpInstance(math, dtype)    \
    UnaryOp<CUDAUnaryFunctor<math<dtype>, dtype>, dtype> 
#define CudaBinaryOpInstance(math, dtype)   \
    BinaryOp<CUDABinaryFunctor<math<dtype>, dtype>, dtype> 
#define CudaBinaryScalarOpInstance(math, dtype)   \
    BinaryOp<CUDABinaryScalarFunctor<math<dtype>, dtype>, dtype> 

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

} //namespace backend
