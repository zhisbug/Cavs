#ifndef CAVS_BACKEND_OPS_IMPL_ELEMENTWISE_COMMON_H_
#define CAVS_BACKEND_OPS_IMPL_ELEMENTWISE_COMMON_H_

#include "cavs/backend/op_impl.h"
#include "cavs/midend/allocator.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/proto/tensor_shape.pb.h"

using ::midend::Tensor;

namespace backend {

template <typename FUNCTOR, typename T>//mathop, dtype
class UnaryOp : public OpImpl {
 public:
  explicit UnaryOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    const Tensor& inp = context->Input(0);
    //inp.DebugNumerical<T>();
    Tensor* out = context->Output(0);
    FUNCTOR::Compute(out->mutable_data<T>(), out->count(), 
        inp.data<T>(), inp.count());
    //out->DebugNumerical<T>();
  }
};

template <typename FUNCTOR, typename T>
class BinaryOp : public OpImpl {
 public:
  explicit BinaryOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    const Tensor& inp0 = context->Input(0);
    const Tensor& inp1 = context->Input(1);
    //inp0.DebugNumerical<T>();
    //inp1.DebugNumerical<T>();
    Tensor* out = context->Output(0);
    FUNCTOR::Compute(out->mutable_data<T>(), out->count(), 
        inp0.data<T>(), inp0.count(), inp1.data<T>(), inp1.count());
    //out->DebugNumerical<T>();
  }
};

} //namespace backend

#endif
