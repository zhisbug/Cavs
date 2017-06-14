#include "cavs/backend/op_decl.h"
#include "cavs/util/op_def_builder.h"

using std::vector;

namespace backend {

class SliceOpDecl : public OpDecl {
 public:
  SliceOpDecl(const OpDef& def) : OpDecl(def),
    split_(-1), index_(-1), offset_(-1), stride_(-1) {
    //CHECK(GetSingleArg<bool>(op_def_, "ShareMemory"));
    if (GetSingleArg(def, "Split", 0) != 0) {
      split_ = GetSingleArg<int>(def, "Split"); 
      index_ = GetSingleArg<int>(def, "Index"); 
      CHECK(split_ > 0);
      CHECK(index_ >= 0);
    }else {
      offset_ = GetSingleArg<int>(def, "Offset");
      stride_ = GetSingleArg<int>(def, "Stride");
      CHECK(offset_ >= 0);
      CHECK(stride_ > 0);
    }
  }
  
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->size() == 0);
    OpDef reshape;
    OpDefBuilder("Accumulate")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .AttrSingle("Offset", offset_)
      .AttrSingle("Stride", stride_)
      .Device(op_def_)
      .Finalize(&reshape);
    grad->push_back(reshape);
  }

  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 1); 
    CHECK(out_shape->empty());
    int dims = 1;
    for (auto d : inputs[0].dim()) dims *= d;
    TensorShapeDef sdef;
    if (offset_ < 0) {
      CHECK(dims % split_ == 0);
      stride_ = dims / split_;
      offset_ = dims / split_ * index_;
    }
    int count = stride_ - offset_;
    CHECK(count <= dims);
    sdef.add_dim(count);

    out_shape->push_back(sdef);
  }

 private:
  int offset_;
  int stride_;
  int split_;
  int index_;
};

REGISTER_OP_DECL_BUILDER("Slice", SliceOpDecl);

} //namespace backend
