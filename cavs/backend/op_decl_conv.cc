#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"
#include "cavs/util/op_util.h"

using std::vector;

namespace backend {

class ConvOpDecl : public OpDecl{
 public:
  ConvOpDecl(const OpDef& def) : OpDecl(def) {};
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK_NOTNULL(grad);
    CHECK(grad->size() == 0);
    CHECK(op_def_.input_size() == 3);
    CHECK(op_def_.output_size() == 1);
    OpDef conv_grad;
    OpDefBuilder("ConvGrad")
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(0))
      .Input(op_def_.input(1))
      .Input(op_def_.input(2))
      .Output(GetGradientName(op_def_.input(1)))
      .Output(GetGradientName(op_def_.input(2)))
      .Output(GetGradientName(op_def_.input(0)))
      .Device(op_def_)
      .Finalize(&conv_grad);
    grad->push_back(std::move(conv_grad));
  }
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 3);
    //Currently, we assume the image layout is N, C, H, W
    //Input dim: N, C, H, W
    //Variable dim: K, C, H, W
    CHECK(inputs[0].dim_size() == 4);
    CHECK(inputs[1].dim_size() == 4);
    CHECK(inputs[0].dim(1) == inputs[1].dim(1));
    //LOG(INFO) << inputs[0].DebugString() << inputs[1].DebugString();
    CHECK(inputs[1].dim(2) == inputs[1].dim(3));//kernel is square
    int pad = GetSingleArg<int>(op_def_, "Pad", 0);
    int stride = GetSingleArg<int>(op_def_, "Stride", 1);
    int N = inputs[0].dim(0);
    int C = inputs[1].dim(0);
    int H = 1 + (inputs[0].dim(2) + 2*pad -inputs[1].dim(2)) /stride;
    int W = 1 + (inputs[0].dim(3) + 2*pad -inputs[1].dim(3)) /stride;
    //LOG(INFO) << pad << "\t" << stride << "\t"
              //<< N << "\t" << C << "\t" << H << "\t" << W;
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0).add_dim(N);
    out_shape->at(0).add_dim(C);
    out_shape->at(0).add_dim(H);
    out_shape->at(0).add_dim(W);
    //LOG(INFO) << out_shape->at(0).DebugString();
  };
};

class ConvGradOpDecl : public OpDecl{
 public:
  ConvGradOpDecl(const OpDef& def) : OpDecl(def) {};
  void MakeGradient(vector<OpDef>* grad) override {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 4);
    CHECK(inputs[0].dim_size() == 4);
    CHECK(inputs[1].dim_size() == 4);
    CHECK(inputs[0].dim(1) == inputs[2].dim(0))
      << inputs[0].DebugString() << inputs[2].DebugString();
    CHECK(inputs[1].dim(1) == inputs[2].dim(1))
      << inputs[1].DebugString() << inputs[2].DebugString();
    out_shape->resize(3);
    out_shape->at(0) = inputs[2];
    out_shape->at(1) = inputs[3];
    out_shape->at(2) = inputs[1];
  };
};

REGISTER_OP_DECL_BUILDER("Conv", ConvOpDecl);
REGISTER_OP_DECL_BUILDER("ConvGrad", ConvGradOpDecl);

} //namespace backend
