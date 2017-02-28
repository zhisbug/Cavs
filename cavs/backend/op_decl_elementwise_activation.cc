#include "cavs/backend/op_decl_elementwise.h"
#include "cavs/backend/op_def_builder.h"

using std::vector;

namespace backend {

class ReluOpDecl : public UnaryOpDecl {
 public:
  explicit ReluOpDecl(const OpDef& def)
    : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 1);
    CHECK(op_def_.output_size() == 1);
    OpDef relu_grad;
    OpDefBuilder("ReluGrad")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .Device(op_def_)
      .Finalize(&relu_grad);
    grad->push_back(std::move(relu_grad));
  }
};

class ReluGradOpDecl : public UnaryOpDecl {
 public:
  explicit ReluGradOpDecl(const OpDef& def)
    : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {}
};

REGISTER_OP_DECL_BUILDER("Relu", ReluOpDecl);
REGISTER_OP_DECL_BUILDER("ReluGrad", ReluGradOpDecl);

} //namespace backend
