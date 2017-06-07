#include "cavs/backend/op_decl.h"
#include "cavs/util/op_def_builder.h"

using std::vector;

namespace backend {

class FullyConnectedOpDecl : public OpDecl {
 public:
  FullyConnectedOpDecl(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 3) << inputs.size();
    CHECK(inputs[0].dim_size() == 2);
    CHECK(inputs[1].dim_size() == 2);
    int batchN = inputs[0].dim(0);
    int K = inputs[0].dim(1);
    int Out = inputs[1].dim(0);
    CHECK(K == inputs[1].dim(1)) << op_def_.DebugString();
    CHECK(inputs[2].dim_size() == 2);
    CHECK(inputs[2].dim(0) == 1);
    CHECK(inputs[2].dim(1) == inputs[1].dim(0));

    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0).add_dim(batchN);
    out_shape->at(0).add_dim(Out);
  }

  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 3);
    CHECK(op_def_.output_size() == 1);
    OpDef fc_grad;
    OpDefBuilder(GetGradientName("FullyConnected"))
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(0))//X
      .Input(op_def_.input(1))//Weight
      .Input(op_def_.input(2))//Bias
      .Output(GetGradientName(op_def_.input(1)))//dWeight
      .Output(GetGradientName(op_def_.input(2)))//dBias
      .Output(GetGradientName(op_def_.input(0)))//dX
      .Device(op_def_)
      .Finalize(&fc_grad);
    grad->push_back(std::move(fc_grad));
  }
};

class FullyConnectedGradOpDecl : public OpDecl {
 public:
  FullyConnectedGradOpDecl(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 4) << inputs.size();
    out_shape->resize(3);
    out_shape->at(0) = inputs[2];
    out_shape->at(1) = inputs[3];
    out_shape->at(2) = inputs[1];
  }
};

REGISTER_OP_DECL_BUILDER("FullyConnected", FullyConnectedOpDecl);
REGISTER_OP_DECL_BUILDER(GetGradientName("FullyConnected"), FullyConnectedGradOpDecl);

} //namespace backend
