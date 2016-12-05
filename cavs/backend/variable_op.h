#ifndef CAVS_BACKEND_VARIABLE_OP_H
#define CAVS_BACKEND_VARIABLE_OP_H

#include "cavs/midend/op.h"
#include "cavs/midend/tensor.h"

namespace cavs {

template <typename FUNCTOR, typename T>//fillop, dtype
class VariableOp : public Op {
 public:
  explicit VariableOp(const OpDef& def) : Op(def) {}

  void Compute(OpContext* context) override {
    LOG(INFO) << "here";
    Tensor* out = context->Output(0);
    LOG(INFO) << "here";
    FUNCTOR::Compute(out->mutable_data<T>(), out->count());
  }
};

} //namespace cavs

#endif
