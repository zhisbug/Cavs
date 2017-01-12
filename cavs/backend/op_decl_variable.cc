#include "cavs/backend/op_decl.h"

using std::vector;

namespace backend {

class VariableOpDecl : public OpDecl{
 public:
  VariableOpDecl(const OpDef& def) : OpDecl(def) {};
  void MakeGradient(vector<OpDef>* grad) override {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override;
};

void VariableOpDecl::ShapeInference(
    vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) {
  out_shape->resize(1);
  out_shape->at(0).clear_dim();
  out_shape->at(0) = op_def_.shape(0);
}

REGISTER_OP_DECL_BUILDER("Variable", VariableOpDecl);

} //namespace backend
