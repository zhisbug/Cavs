#include "cavs/backend/op_decl_elementwise.h"

using std::vector;

namespace backend {

class ReluOpDecl : public UnaryOpDecl {
 public:
  explicit ReluOpDecl(const OpDef& def)
    : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    MakeGradientUnary(grad);
  }
};

REGISTER_OP_DECL_BUILDER("Relu", ReluOpDecl);
//ReluGrad operator does not need a gradient further
REGISTER_OP_DECL_BUILDER(GetGradientName("Relu"), UnaryOpDecl);

} //namespace backend
