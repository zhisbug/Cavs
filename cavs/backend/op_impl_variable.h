#ifndef CAVS_BACKEND_OP_IMPL_VARIABLE_H_
#define CAVS_BACKEND_OP_IMPL_VARIABLE_H_

#include "cavs/backend/op_impl.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/op_context.h"
#include "cavs/proto/tensor_shape.pb.h"

namespace backend {

using ::midend::OpContext;
using ::midend::Tensor;

template <typename FILLFUNCTOR, typename T>//fillop, dtype
class VariableOpImpl : public OpImpl {
 public:
  explicit VariableOpImpl(const OpDef& def)
    : OpImpl(def), initialized_(false) {
  }

  void Compute(OpContext* context) override {
    if (!initialized_) {
      Tensor* out = context->Output(0);
      FILLFUNCTOR(op_def_).Compute(out->mutable_data<T>(), out->count());
      initialized_ = true;
    }
  }
 private:
  bool initialized_;
};

} //namespace backend

#endif
