#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"

using std::vector;

namespace backend {

class ReduceMeanOp: public OpDecl {
 public:
  explicit ReduceMeanOp(const OpDef& def) : OpDecl(def) {}
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

REGISTER_OP_DECL_BUILDER("Reduce_mean", ReduceMeanOp);

} //namespace backend
