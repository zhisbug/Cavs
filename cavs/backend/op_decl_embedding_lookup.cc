#include "cavs/backend/op_decl.h"
#include "cavs/util/op_util.h"
#include "cavs/util/op_def_builder.h"

using std::vector;

namespace backend {

class EmbeddingLookupOpDecl : public OpDecl{
 public:
  EmbeddingLookupOpDecl(const OpDef& def) : OpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK_NOTNULL(grad);
    CHECK(grad->size() == 0);
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    OpDef embed_grad;
    OpDefBuilder(GetGradientName("EmbeddingLookup"))
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(0))
      .Input(op_def_.input(1))
      .Output(GetGradientName(op_def_.input(1)))
      .Output(GetGradientName(op_def_.input(0)))
      .Device(op_def_)
      .Finalize(&embed_grad);
    grad->push_back(std::move(embed_grad));
  }
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 2);
    int vocabulary_size = inputs[1].dim(0);
    int embedding_size  = inputs[1].dim(1);

    out_shape->resize(1);
    for (int i = 0; i < inputs[0].dim_size(); i++) {
      out_shape->at(0).add_dim(inputs[0].dim(i));
    }
    out_shape->at(0).add_dim(embedding_size);
  };
};

class EmbeddingLookupGradOpDecl : public OpDecl{
 public:
  EmbeddingLookupGradOpDecl(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 3);
    CHECK(out_shape->empty());
    out_shape->push_back(inputs[2]);
    out_shape->push_back(inputs[1]);
  };
};

REGISTER_OP_DECL_BUILDER("EmbeddingLookup", EmbeddingLookupOpDecl);
REGISTER_OP_DECL_BUILDER(GetGradientName("EmbeddingLookup"), EmbeddingLookupGradOpDecl);

} //namespace backend

