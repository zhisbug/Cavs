#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"

using std::vector;

namespace backend {

class MatMulOpDecl : public OpDecl {
 public:
  MatMulOpDecl(const OpDef& def) : OpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    OpDef mul_def_0;
    OpDefBuilder("MatMul")
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(1))
      .Output(GetGradientName(op_def_.input(0)))
      .Device(op_def_)
      .Finalize(&mul_def_0);
    grad->push_back(std::move(mul_def_0));
    OpDef mul_def_1;
    OpDefBuilder("MatMul")
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(0))
      .Output(GetGradientName(op_def_.input(1)))
      .Device(op_def_)
      .Finalize(&mul_def_1);
    grad->push_back(std::move(mul_def_1));
  }
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 2) << inputs.size();
    CHECK(inputs[0].dim_size() == 2);
    CHECK(inputs[1].dim_size() == 2);
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0).add_dim(inputs[0].dim(0));
    out_shape->at(0).add_dim(inputs[1].dim(1));
  }
};

REGISTER_OP_DECL_BUILDER("MatMul", MatMulOpDecl);

}; //namespace backend
