#ifndef CAVS_BACKEND_PLACEHOLDER_OP_H
#define CAVS_BACKEND_PLACEHOLDER_OP_H

#include "cavs/midend/op.h"
#include "cavs/midend/tensor.h"

namespace cavs {

template <typename FUNCTOR, typename T>//copyop, dtype
class PlaceholderOp : public Op {
 public:
  explicit PlaceholderOp(const OpDef& def) : Op(def) {}

  void Compute(OpContext* context) override {
    LOG(INFO) << "here";
    const Tensor* inp = context->Input(0);
    LOG(INFO) << "here";
    Tensor* out = context->Output(0);
    LOG(INFO) << "here";
    FUNCTOR::Compute(out->mutable_data<T>(), inp->data<T>(), out->count());
    LOG(INFO) << "here";
  }
};

} //namespace cavs

#endif
