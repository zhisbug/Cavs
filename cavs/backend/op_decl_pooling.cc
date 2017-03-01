#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"
#include "cavs/util/op_util.h"

using std::vector;

namespace backend {

class PoolingOpDecl : public OpDecl{
 public:
  PoolingOpDecl(const OpDef& def) : OpDecl(def) {};
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->size() == 0);
    CHECK(op_def_.input_size() == 1)
      << op_def_.DebugString();
    CHECK(op_def_.output_size() == 1)
      << op_def_.DebugString();
    OpDef conv_grad;
    OpDefBuilder("PoolingGrad")
      .Input(op_def_.output(0))
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(0))
      .Output(GetGradientName(op_def_.input(0)))
      .Device(op_def_)
      .Attr(op_def_)
      .Finalize(&conv_grad);
    //LOG(INFO) << op_def_.DebugString();
    //LOG(INFO) << conv_grad.DebugString();
    grad->push_back(std::move(conv_grad));
  }
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 1);
    //Currently, we assume the image layout is N, C, H, W
    //Input dim: N, C, H, W
    CHECK(inputs[0].dim_size() == 4);
    int height_window = GetSingleArg<int>(op_def_, "HightWindow");
    int width_window = GetSingleArg<int>(op_def_, "WidthWindow");
    //Currently, the strided and padding is hard coded
    int height_stride = height_window;
    int width_stride = width_window;
    int height_padding = 0;
    int width_padding = 0;
    int N = inputs[0].dim(0);
    int C = inputs[0].dim(1);
    int H = 1 + (inputs[0].dim(2) + 2*height_padding - height_window) / height_stride;
    int W = 1 + (inputs[0].dim(3) + 2*height_padding - height_window) / height_stride;
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0).add_dim(N);
    out_shape->at(0).add_dim(C);
    out_shape->at(0).add_dim(H);
    out_shape->at(0).add_dim(W);
  };
};

class PoolingGradOpDecl : public OpDecl{
 public:
  PoolingGradOpDecl(const OpDef& def) : OpDecl(def) {};
  void MakeGradient(vector<OpDef>* grad) override {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 3);
    CHECK(inputs[0].dim_size() == 4);//y
    CHECK(inputs[1].dim_size() == 4);//dy
    CHECK(inputs[2].dim_size() == 4);//x
    CHECK(inputs[0].dim(0) == inputs[1].dim(0));
    CHECK(inputs[0].dim(1) == inputs[1].dim(1));
    CHECK(inputs[0].dim(2) == inputs[1].dim(2));
    CHECK(inputs[0].dim(3) == inputs[1].dim(3));
    CHECK(inputs[0].dim(0) == inputs[2].dim(0));
    CHECK(inputs[0].dim(1) == inputs[2].dim(1));
    out_shape->resize(1);
    out_shape->at(0) = inputs[2];
  };
};

REGISTER_OP_DECL_BUILDER("Pooling", PoolingOpDecl);
REGISTER_OP_DECL_BUILDER("PoolingGrad", PoolingGradOpDecl);

} //namespace backend
