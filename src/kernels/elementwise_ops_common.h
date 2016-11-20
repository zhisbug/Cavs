#ifndef ELEMENTWISE_OPS_COMMON_H_
#define ELEMENTWISE_OPS_COMMON_H_

#include "op.h"

template <typename Functor>
class UnaryOp : public Op {
 public:
  explicit UnaryOp(const OpDef& def, Session* s) : Op(def, s) {}
  void Compute() override {
    const Tensor& inp = Input(0);
    Tensor* out = Output(0);
    Functor()(out->flat(), inp.flat());
  }
};

template <typename Functor>
class BinaryOp : public Op {
 public:
  explicit UnaryOp(const OpDef& def, Session *s) : Op(def, s) {}
  void Compute() override {
    const Tensor& inp0 = Input(0);
    const Tensor& inp1 = Input(0);
    Tensor* out = Output(0);
    Functor()(out->flat(), inp0.flat(), inp1.flat());
  }
};



#endif
