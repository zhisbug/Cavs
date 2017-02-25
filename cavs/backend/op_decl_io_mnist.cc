#include "cavs/backend/op_decl.h"
#include "cavs/util/op_util.h"

using std::vector;

namespace backend {

class MnistInputOpDecl : public OpDecl{
 public:
  MnistInputOpDecl(const OpDef& def) : OpDecl(def) {};
  void MakeGradient(vector<OpDef>* grad) override {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 0);
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    int batch = GetSingleArg<int>(op_def_, "Batch");
    out_shape->at(0).add_dim(batch);
    out_shape->at(0).add_dim(1);
    out_shape->at(0).add_dim(28);
    out_shape->at(0).add_dim(28);
  };
};

REGISTER_OP_DECL_BUILDER("MnistInput", MnistInputOpDecl);

} //namespace backend
