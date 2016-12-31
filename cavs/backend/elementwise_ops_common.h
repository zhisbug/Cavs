#ifndef CAVS_KERNEL_ELEMENTWISE_OPS_COMMON_H_
#define CAVS_KERNEL_ELEMENTWISE_OPS_COMMON_H_

#include "cavs/midend/op.h"
#include "cavs/midend/op_def.pb.h"
#include "cavs/midend/tensor_shape.pb.h"
#include "cavs/midend/allocator.h"

namespace backend {

using ::midend::Op;
using ::midend::OpContext;
using ::midend::OpDef;
using ::midend::Tensor;
using ::midend::TensorShapeDef;

template <typename FUNCTOR, typename T>//mathop, dtype
class UnaryOp : public Op {
 public:
  explicit UnaryOp(const OpDef& def) : Op(def) {
    //const Tensor* inp = Input(0);
    //Tensor* out = Output(0);
    //out->Reshape(GetAllocator(def), def.out_type(), inp->shape());
  }
  void Compute(OpContext* context) override {
    const Tensor& inp = context->Input(0);
    Tensor* out = context->Output(0);
    //test_func(out->mutable_data<T>(), inp->data<T>(), inp->count());
    FUNCTOR::Compute(out->mutable_data<T>(), inp.data<T>(), inp.count());
  }

  static void ShapeInference(TensorShapeDef* shape,
    const OpDef& def, const vector<const TensorShapeDef*>& inputs) {
    CHECK(inputs.size() == 1);
    shape->clear_dim();
    for (auto dim : inputs[0]->dim())
      shape->add_dim(dim);
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
    const Tensor& inp0 = context->Input(0);
    const Tensor& inp1 = context->Input(1);
    Tensor* out = context->Output(0);
    FUNCTOR::Compute(out->mutable_data<T>(), inp0.data<T>(), inp1.data<T>(),
            inp0.count());
  }

  static void ShapeInference(TensorShapeDef* shape,
    const OpDef& def, const vector<const TensorShapeDef*>& inputs) {
    CHECK(inputs.size() == 2);
    shape->clear_dim();
    for (auto dim : inputs[0]->dim())
      shape->add_dim(dim);
  }
};

} //namespace backend

#endif
