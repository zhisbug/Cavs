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
    //do nothing now
  }
};

} //namespace cavs

#endif
