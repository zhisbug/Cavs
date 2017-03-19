#include "cavs/backend/op_decl.h"

using std::vector;

namespace backend {

class ConstOpDecl : public OpDecl {
 public:
  ConstOpDecl(const OpDef& def) : OpDecl(def) {};
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override;
};

void ConstOpDecl::ShapeInference(
    vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) {
  CHECK(op_def_.shape_size() == 1);
  CHECK(op_def_.shape(0).dim_size() >= 1);
  out_shape->resize(1);
  out_shape->at(0).clear_dim();
  out_shape->at(0) = op_def_.shape(0);
}

REGISTER_OP_DECL_BUILDER("ConstOp", ConstOpDecl);

} //namespace backend
