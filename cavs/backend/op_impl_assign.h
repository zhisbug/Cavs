#ifndef CAVS_BACKEND_OP_IMPL_ASSIGN_H_
#define CAVS_BACKEND_OP_IMPL_ASSIGN_H_

#include "cavs/backend/op_impl.h"
#include "cavs/midend/allocator.h"
#include "cavs/midend/tensor.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/proto/tensor_shape.pb.h"

namespace backend {

using ::midend::Tensor;

class OpImplAssign : public OpImpl {
 public:
  explicit OpImplAssign(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    const Tensor& inp = context->Input(0); 
    Tensor* out = context->Output(0);
    *out = inp;
  }
};

}

#endif
