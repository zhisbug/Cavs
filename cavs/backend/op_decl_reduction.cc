#include "cavs/backend/op_decl.h"
#include "cavs/util/op_def_builder.h"

using std::vector;

namespace backend {

class ReduceOp: public OpDecl {
 public:
  explicit ReduceOp(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 1);
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0).add_dim(1);
  }

  void MakeGradient(vector<OpDef>* grad) override {
    OpDef fill_def;
    OpDefBuilder("Fill")
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(0))
      .Output(GetGradientName(op_def_.input(0)))
      .Device(op_def_)
      .Finalize(&fill_def);
    grad->push_back(std::move(fill_def));
  }
};

class ArgmaxOp: public OpDecl {
 public:
  explicit ArgmaxOp(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    int axis = GetSingleArg<int>(op_def_, "Axis");
    CHECK(inputs.size() == 1);
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    for (int i = 0; i < axis; i++)
      out_shape->at(0).add_dim(inputs[0].dim(i));
    out_shape->at(0).add_dim(1);
  }

  //Not decided yet
  void MakeGradient(vector<OpDef>* grad) override {
  }
};

REGISTER_OP_DECL_BUILDER("Reduce_sum", ReduceOp);
REGISTER_OP_DECL_BUILDER("Reduce_mean", ReduceOp);
REGISTER_OP_DECL_BUILDER("Argmax", ArgmaxOp);

} //namespace backend
