#include "cavs/backend/op_decl_elementwise.h"

using std::vector;

namespace backend {

class ReluOpDecl : public UnaryOpDecl {
 public:
  explicit ReluOpDecl(const OpDef& def)
    : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->size() == 0);
    CHECK(op_def_.input_size() == 1);
    CHECK(op_def_.output_size() == 1);
    OpDef grad_def;
    OpDefBuilder(GetGradientName(op_def_.name()))
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.output(0))
      .Input(op_def_.input(0))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .Device(op_def_)
      .Finalize(&grad_def);
    grad->push_back(std::move(grad_def));
  }
};

REGISTER_OP_DECL_BUILDER("Relu", ReluOpDecl);
//ReluGrad operator does not need a gradient further
REGISTER_OP_DECL_BUILDER(GetGradientName("Relu"), TernaryOpDecl);

} //namespace backend
