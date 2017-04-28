#include "cavs/backend/op_decl_elementwise.h"

using std::vector;

namespace backend {

//Softmax is an elementwise operation, 
//but is different from unary/binary operator,
//Because it has two inputs, one of which(label)
//is not used in forward but is used in backward.
class SoftmaxOpDecl : public OpDecl {
 public:
  explicit SoftmaxOpDecl(const OpDef& def)
    : OpDecl(def) {}
  void ShapeInference(std::vector<TensorShapeDef>* out_shape,
    const std::vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 2);
    //images and labels share the same N(batch size);
    CHECK(inputs[0].dim(0) == inputs[1].dim(0));
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0) = inputs[0];
  }
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->size() == 0);
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    OpDef grad_def;
    OpDefBuilder(GetGradientName(op_def_.name()))
      .Input(op_def_.output(0))
      .Input(op_def_.input(1))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .Device(op_def_)
      .Finalize(&grad_def);
    grad->push_back(std::move(grad_def));
  }
};

class SoftmaxGradOpDecl : public OpDecl {
 public:
  explicit SoftmaxGradOpDecl(const OpDef& def)
    : OpDecl(def) {}
  void ShapeInference(std::vector<TensorShapeDef>* out_shape,
    const std::vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 2);
    //images and labels share the same N(batch size);
    CHECK(inputs[0].dim(0) == inputs[1].dim(0));
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0) = inputs[0];
  }
};

REGISTER_OP_DECL_BUILDER("SoftmaxEntropyLogits", SoftmaxOpDecl);
//Softmax gradient operator does not need a gradient further
REGISTER_OP_DECL_BUILDER(GetGradientName("SoftmaxEntropyLogits"),
    SoftmaxGradOpDecl);

} //namespace backend
