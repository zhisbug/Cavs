#include "cavs/backend/op_impl_variable.h"
#include "cavs/backend/functor_filler.cuh"
#include "cavs/backend/functor_elementwise.h"

namespace backend {

/*REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU").Label("ConstantFiller"),*/
    /*VariableOpImpl<CudaConstantFiller<math::Assign<float>, float>, float>);*/
/*REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU").Label("UniformRandom"),*/
    /*VariableOpImpl<CudaFiller<UniformNormalizer<float>, float>, float>);*/
/*REGISTER_OP_IMPL_BUILDER(Key("DDV").Device("GPU").Label("ConstantFiller"),*/
    /*DDVOpImpl<CudaConstantFiller<math::Assign<float>, float>, float>);*/
/*REGISTER_OP_IMPL_BUILDER(Key("DDV").Device("GPU").Label("UniformRandom"),*/
    /*DDVOpImpl<Filler<UniformNormalizer<float>, float>, float>);*/

REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU").Label("ConstantFiller"),
    VariableOpImpl<CudaFiller<ConstantFiller<float>, float>, float>);
REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU").Label("UniformNormalizer"),
    VariableOpImpl<CudaFiller<UniformRandomNormalized<float>, float>, float>);
REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU").Label("Xavier"),
    VariableOpImpl<CudaFiller<Xavier<float>, float>, float>);
/*REGISTER_OP_IMPL_BUILDER(Key("Variable").Device("GPU").Label("NormalRandom"),*/
    /*VariableOpImpl<CudaFiller<NormalRandom<float>, float>, float>);*/
REGISTER_OP_IMPL_BUILDER(Key("VariableMPI").Device("GPU").Label("ConstantFiller"),
    VariableOpImpl<CudaFiller<ConstantFiller<float>, float>, float, MPIBcastFunctor<float>>);
REGISTER_OP_IMPL_BUILDER(Key("VariableMPI").Device("GPU").Label("UniformNormalizer"),
    VariableOpImpl<CudaFiller<UniformRandomNormalized<float>, float>, float, MPIBcastFunctor<float>>);
REGISTER_OP_IMPL_BUILDER(Key("VariableMPI").Device("GPU").Label("Xavier"),
    VariableOpImpl<CudaFiller<Xavier<float>, float>, float, MPIBcastFunctor<float>>);
/*REGISTER_OP_IMPL_BUILDER(Key("VariableMPI").Device("GPU").Label("NormalRandom"),*/
    /*VariableOpImpl<CudaFiller<NormalRandom<float>, float>, float, MPIBcastFunctor<float>>);*/

REGISTER_OP_IMPL_BUILDER(Key("DDV").Device("GPU").Label("UniformNormalizer"),
    DDVOpImpl<UniformRandomNormalized<float>, float, false>);
REGISTER_OP_IMPL_BUILDER(Key("DDVMPI").Device("GPU").Label("UniformNormalizer"),
    DDVOpImpl<UniformRandomNormalized<float>, float, true>);

} //namespace backend
