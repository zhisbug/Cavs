#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
#include "cavs/backend/functor_elementwise.h"
#include "cavs/backend/op_impl_elementwise.cuh"

namespace backend {

using ::midend::Tensor;

//absolute value mean
template <typename T>
class AmeanOpCublas : public OpImpl {
 public:
  explicit AmeanOpCublas(const OpDef& def)
    : OpImpl(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
void AmeanOpCublas<T>::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  Tensor* y = context->Output(0);
  CHECK(1 == y->count());
  int N = x.count();
  CHECK(N > 0);

  //x.DebugNumerical<T>();
  AsumCublasWrapper<T>(
      N, x.data<T>(), y->mutable_data<T>());
  CUDABinaryConstScalarFunctor<math::Div<T>, T>:: Compute(
      y->mutable_data<T>(), y->count(), 
      y->mutable_data<T>(), y->count(), 
      x.count());
  //y->DebugNumerical<T>();
}

REGISTER_OP_IMPL_BUILDER(Key("Reduce_mean").Device("GPU"),
    AmeanOpCublas<float>);

} //namespace backend

