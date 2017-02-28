#include "cavs/backend/op_decl_elementwise.h"

using std::vector;

namespace backend {

class SoftmaxOpDecl : public UnaryOpDecl {
 public:
  explicit SoftmaxOpDecl(const OpDef& def)
    : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    MakeGradientUnary(grad);
  }
};

REGISTER_OP_DECL_BUILDER("SoftmaxEntropyLogits", SoftmaxOpDecl);
//Softmax gradient operator does not need a gradient further
REGISTER_OP_DECL_BUILDER(GetGradientName("SoftmaxEntropyLogits"), UnaryOpDecl);

} //namespace backend
