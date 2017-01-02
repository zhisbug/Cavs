#ifndef CAVS_BACKEND_PLACEHOLDER_OP_H
#define CAVS_BACKEND_PLACEHOLDER_OP_H

#include "cavs/midend/op.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/tensor_shape.pb.h"

namespace backend {

using ::midend::Op;
using ::midend::OpContext;
using ::midend::OpDef;
using ::midend::Tensor;
using ::midend::TensorShapeDef;

template <typename FUNCTOR, typename T>//copyop, dtype
class PlaceholderOp : public Op {
 public:
  explicit PlaceholderOp(const OpDef& def) : Op(def) {}

  void Compute(OpContext* context) override {
    //do nothing now
  }

  static void ShapeInference(TensorShapeDef* shape,
    const OpDef& def, const vector<const TensorShapeDef*>& inputs) {
    shape->clear_dim();  
    for (auto& attr : def.attr()) {
      if (attr.name() == "shape") {
        CHECK(attr.value().has_shape());
        *shape = attr.value().shape();
      }
    }
  }
};

} //namespace cavs

#endif
