#include "cavs/backend/op_impl_variable.h"
#include "cavs/backend/functor_filler.cuh"
#include "cavs/backend/functor_elementwise.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU").Label("ConstantFiller"),
    VariableOpImpl<CudaConstantFiller<math::Assign<float>, float>, float>);
REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU").Label("UniformRandom"),
    VariableOpImpl<CudaFiller<UniformNormalizer<float>, float>, float>);
/*REGISTER_OP_IMPL_BUILDER(Key("DDV").Device("GPU").Label("ConstantFiller"),*/
    /*DDVOpImpl<CudaConstantFiller<math::Assign<float>, float>, float>);*/
REGISTER_OP_IMPL_BUILDER(Key("DDV").Device("GPU").Label("UniformRandom"),
    DDVOpImpl<Filler<UniformNormalizer<float>, float>, float>);

} //namespace backend
