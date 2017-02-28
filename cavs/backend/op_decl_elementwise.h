#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"
#include "cavs/util/op_util.h"

namespace backend {

class UnaryOpDecl : public OpDecl {
 public:
  explicit UnaryOpDecl(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(std::vector<TensorShapeDef>* out_shape,
    const std::vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 1);
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0) = inputs[0];
  }

 protected:
  void MakeGradientUnary(std::vector<OpDef>* grad) {
    CHECK(grad->size() == 0);
    CHECK(op_def_.input_size() == 1);
    CHECK(op_def_.output_size() == 1);
    OpDef grad_def;
    OpDefBuilder(GetGradientName(op_def_.name()))
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .Device(op_def_)
      .Finalize(&grad_def);
    grad->push_back(std::move(grad_def));
  }
};

class BinaryOpDecl : public OpDecl {
 public:
  explicit BinaryOpDecl(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(std::vector<TensorShapeDef>* out_shape,
    const std::vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 2) << inputs.size();
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0) = inputs[0];
  }
};

} //namespace backend
