#include "cavs/backend/op_decl.h"

using std::vector;

namespace backend {

class VariableOpDecl : public OpDecl{
 public:
  VariableOpDecl(const OpDef& def) : OpDecl(def) {};
  //void MakeGradient(vector<OpDef>* grad) override {}
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

class DDVOpDecl : public OpDecl{
 public:
  DDVOpDecl(const OpDef& def) : OpDecl(def) {};
  //void MakeGradient(vector<OpDef>* grad) override {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override;
};

void DDVOpDecl::ShapeInference(
    vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) {
  CHECK(out_shape);
  CHECK(out_shape->empty());
  CHECK(op_def_.shape_size() == 0);
  //CHECK(op_def_.shape(0).dim_size() == 0);
  int batch = GetSingleArg<int>(op_def_, "Batch");
  const std::vector<int>& shape = GetListArg<int>(op_def_, "Shape");
  CHECK(!shape.empty());
  CHECK(shape.size() > 1);
  out_shape->resize(1);
  out_shape->at(0).add_dim(batch);
  for (int i = 1; i < shape.size(); i++) {
    out_shape->at(0).add_dim(shape[i]);
  }
}

REGISTER_OP_DECL_BUILDER("Variable", VariableOpDecl);
REGISTER_OP_DECL_BUILDER("DDV", DDVOpDecl);

} //namespace backend
