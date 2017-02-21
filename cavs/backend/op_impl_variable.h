#ifndef CAVS_BACKEND_OP_IMPL_VARIABLE_H_
#define CAVS_BACKEND_OP_IMPL_VARIABLE_H_

#include "cavs/backend/op_impl.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/op_context.h"
#include "cavs/proto/tensor_shape.pb.h"

namespace backend {

using ::midend::OpContext;
using ::midend::Tensor;

template <typename FUNCTOR, typename T>//fillop, dtype
class VariableOpImpl : public OpImpl {
 public:
  explicit VariableOpImpl(const OpDef& def)
    : OpImpl(def), initialized(false) {
    init = GetSingleArg<T>("ConstFiller");
  }

  void Compute(OpContext* context) override {
    if (!initialized) {
      Tensor* out = context->Output(0);
      FUNCTOR::Compute(out->mutable_data<T>(), init, out->count());
      initialized = true;
    }
  }
 private:
  T init;
  bool initialized;
};

} //namespace backend

#endif
