#include "cavs/backend/op_decl.h"

using std::vector;

namespace backend {

class PlaceholderOpDecl : public OpDecl {
 public:
  PlaceholderOpDecl(const OpDef& def) : OpDecl(def) {}
  //void MakeGradient(vector<OpDef>* grad) override {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override;
};

void PlaceholderOpDecl::ShapeInference(
    vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) {
  out_shape->resize(1);
  out_shape->at(0).clear_dim();
  out_shape->at(0) = op_def_.shape(0);
}

class DataOpDecl : public OpDecl {
 public:
  DataOpDecl(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override;
};

void DataOpDecl::ShapeInference(
    vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) {
  CHECK(out_shape);
  CHECK(out_shape->empty());
  CHECK(op_def_.shape_size() == 1);
  CHECK(op_def_.shape(0).dim_size() == 0);
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

REGISTER_OP_DECL_BUILDER("Placeholder", PlaceholderOpDecl);
REGISTER_OP_DECL_BUILDER("Data", DataOpDecl);

} //namespace backend
