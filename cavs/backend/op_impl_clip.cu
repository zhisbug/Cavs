#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
#include "cavs/backend/op_impl_elementwise.cuh"
#include "cavs/backend/functor_elementwise.h"

#include <algorithm>

namespace backend {

using ::midend::Tensor;

template <typename T>
class ClipOpImpl : public OpImpl {
 public:
  explicit ClipOpImpl(const OpDef& def) : OpImpl(def) {
    clip_ = GetSingleArg<float>(def, "clip"); 
    CHECK(clip_ > 0);
  }
  void Compute(OpContext* context) override;

 private:
  float clip_;
};

template <typename T>
void ClipOpImpl<T>::Compute(OpContext* context) {
  T sum = 0;
  for (int i = 0; i < context->InputSize(); i++) {
    const Tensor& value = context->Input(i);
    T tmp;
    Nrm2CublasWrapperHost(value.count(), value.data<T>(), &tmp);
    sum += tmp;
  }

  CHECK(sum > 0);
  for (int i = 0; i < context->OutputSize(); i++) {
    const Tensor& in = context->Input(i);
    Tensor* out = context->Output(i);
    CUDABinaryConstScalarFunctor<math::Div<T>, T>::Compute(
        out->mutable_data<T>(), out->count(),
        in.data<T>(), in.count(), 
        clip_/std::max(sum, clip_));
  }
}

REGISTER_OP_IMPL_BUILDER(Key("Clip").Device("GPU"), ClipOpImpl<float>);

} //namespace backend
