#include "cavs/backend/op_decl.h"

namespace backend {

class ConvOpDeclBase : public OpDecl {
 public:
  explicit ConvOpDeclBase(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(vector<TensorShapeDef>* shape,
    const vector<TensorShapeDef>& inputs) override {}
  virtual void MakeGradient(vector<OpDef>* grad) override {}
};

REGISTER_OP_DECL_BUILDER("Conv", ConvOpDeclBase);

//static TensorShape conv_op_shape_inference(OpDef* def) {
  //int pad_h = 0;
  //int pad_w = 0;
  //int stride_h = 1;
  //int stride_w = 1;

  //check
  //int XN = def.
//}

} //namespace backend
