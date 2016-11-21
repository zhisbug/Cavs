#ifndef CAVS_KERNEL_ELEMENTWISE_OPS_COMMON_H_
#define CAVS_KERNEL_ELEMENTWISE_OPS_COMMON_H_

#include "cavs/core/op.h"
#include "cavs/core/op_def.pb.h"

namespace cavs {

template <typename FUNCTOR, typename T>//device, mathop, dtype
class UnaryOp : public Op {
 public:
  explicit UnaryOp(const OpDef& def, Session* s) : Op(def, s) {}
  void Compute() override {
    const Tensor* inp = Input(0);
    Tensor* out = Output(0);
    //test_func(out->mutable_data<T>(), inp->data<T>(), inp->count());
    FUNCTOR::Run(out->mutable_data<T>(), inp->data<T>(), inp->count());
  }
};

template <typename FUNCTOR, typename T>
class BinaryOp : public Op {
 public:
  explicit BinaryOp(const OpDef& def, Session *s) : Op(def, s) {}
  void Compute() override {
    const Tensor* inp0 = Input(0);
    const Tensor* inp1 = Input(1);
    Tensor* out = Output(0);
    FUNCTOR::Run(out->mutable_data<T>(), inp0->data<T>(), inp1->data<T>(),
            inp0->count());
  }
 private:
  FUNCTOR functor_;
};

} //namespace cavs

#endif
