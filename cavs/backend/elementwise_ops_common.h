#ifndef CAVS_KERNEL_ELEMENTWISE_OPS_COMMON_H_
#define CAVS_KERNEL_ELEMENTWISE_OPS_COMMON_H_

#include "cavs/midend/op.h"
#include "cavs/midend/op_def.pb.h"
#include "cavs/midend/allocator.h"

namespace cavs {

template <typename FUNCTOR, typename T>//device, mathop, dtype
class UnaryOp : public Op {
 public:
  explicit UnaryOp(const OpDef& def) : Op(def) {
    //const Tensor* inp = Input(0);
    //Tensor* out = Output(0);
    //out->Reshape(GetAllocator(def), def.out_type(), inp->shape());
  }
  void Compute(OpContext* context) override {
    const Tensor* inp = context->Input(0);
    Tensor* out = context->Output(0);
    //test_func(out->mutable_data<T>(), inp->data<T>(), inp->count());
    FUNCTOR::Run(out->mutable_data<T>(), inp->data<T>(), inp->count());
  }
};

template <typename FUNCTOR, typename T>
class BinaryOp : public Op {
 public:
  explicit BinaryOp(const OpDef& def) : Op(def) {
    //const Tensor* inp = Input(0);
    //Tensor* out = Output(0);
    //out->Reshape(GetAllocator(def), def.out_type(), inp->shape());
  }
  void Compute(OpContext* context) override {
    const Tensor* inp0 = context->Input(0);
    const Tensor* inp1 = context->Input(1);
    Tensor* out = context->Output(0);
    FUNCTOR::Run(out->mutable_data<T>(), inp0->data<T>(), inp1->data<T>(),
            inp0->count());
  }
 private:
  FUNCTOR functor_;
};

} //namespace cavs

#endif
