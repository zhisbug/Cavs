#include "cavs/backend/op_decl.h"
#include "cavs/util/op_util.h"
#include "cavs/util/op_def_builder.h"

using std::vector;

namespace backend {

class LSTMOpDecl : public OpDecl{
 public:
  LSTMOpDecl(const OpDef& def) : OpDecl(def) {};
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK_NOTNULL(grad);
    CHECK(grad->size() == 0);
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    OpDef LSTM_grad;
    OpDefBuilder(GetGradientName("LSTM"))
      .Input(op_def_.output(0))//Y
      .Input(GetGradientName(op_def_.output(0)))//dY
      .Input(op_def_.input(0))//X
      .Input(op_def_.input(1))//W
      .Output(GetGradientName(op_def_.input(0)))//dX
      .Output(GetGradientName(op_def_.input(1)))//dW
      .Attr(op_def_)
      .Device(op_def_)
      .Finalize(&LSTM_grad);
    grad->push_back(std::move(LSTM_grad));
  }
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    const int hidden_size = GetSingleArg<int>(op_def_, "hidden_size");
    CHECK(inputs.size() == 2);
    CHECK(inputs[0].dim_size() == 3);
    const int seq_length = inputs[0].dim(0);
    const int batch      = inputs[0].dim(1);
    //const int input_size = inputs[0].dims(2);
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0).add_dim(seq_length);
    out_shape->at(0).add_dim(batch);
    out_shape->at(0).add_dim(hidden_size);
    VLOG(V_DEBUG) << out_shape->at(0).DebugString();
  };
};

class LSTMGradOpDecl : public OpDecl{
 public:
  LSTMGradOpDecl(const OpDef& def) : OpDecl(def) {};
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 4);
    CHECK(out_shape->empty());
    out_shape->resize(2);
    out_shape->at(0) = inputs[2];
    out_shape->at(1) = inputs[3];
  };
};

REGISTER_OP_DECL_BUILDER("LSTM", LSTMOpDecl);
REGISTER_OP_DECL_BUILDER(GetGradientName("LSTM"), LSTMGradOpDecl);

} //namespace backend
