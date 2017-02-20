#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"

using std::vector;

namespace backend {

class ReductionOpDecl : public OpDecl {
 public:
  explicit ReductionOpDecl(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 1);
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0).add_dim(1);
  }
};

class SquareOpDecl : public ReductionOpDecl {
 public:
  explicit SquareOpDecl(const OpDef& def) : ReductionOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 1);
    CHECK(op_def_.output_size() == 1);
    for (int i = 0; i < op_def_.input_size(); i++) {
      OpDef assign_def;
      OpDefBuilder("Scal")
        .Input(op_def_.input(0))
        .Input(GetGradientName(op_def_.output(0)))
        .Output(GetGradientName(op_def_.input(0)))
        .Shape(op_def_)
        .Attr("alpha", 2.f)
        .Device(op_def_)
        .Finalize(&assign_def);
      grad->push_back(std::move(assign_def));
    }
  }
};

REGISTER_OP_DECL_BUILDER("Square", SquareOpDecl);

} //namespace backend
