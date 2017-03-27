#ifndef CAVS_BACKEND_OP_IMPL_PLACEHOLDER_H_
#define CAVS_BACKEND_OP_IMPL_PLACEHOLDER_H_

#include "cavs/backend/op_impl.h"
#include "cavs/midend/tensor.h"
#include "cavs/proto/tensor_shape.pb.h"

namespace backend {

using ::midend::OpContext;
using ::midend::Tensor;

template <typename FUNCTOR, typename T>//copyop, dtype
class PlaceholderOpImpl : public OpImpl {
 public:
  explicit PlaceholderOpImpl(const OpDef& def) : OpImpl(def) {}

  void Compute(OpContext* context) override {
    //do nothing now
  }
};

template <typename FUNCTOR, typename T>//copyop, dtype
class DataOpImpl : public OpImpl {
 public:
  explicit DataOpImpl(const OpDef& def) : OpImpl(def) {}

  void Compute(OpContext* context) override {
    //do nothing now
  }
};

} //namespace cavs

#endif
