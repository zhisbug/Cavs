#include "cavs/backend/op_decl.h"
#include "cavs/util/op_def_builder.h"

using std::vector;
using std::string;

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
    OpDef slice;
    OpDefBuilder("Accumulate")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .AttrSingle("Offset", offset_)
      .AttrSingle("Stride", stride_)
      .Device(op_def_)
      .Finalize(&slice);
    grad->push_back(slice);
  }

  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 1); 
    CHECK(out_shape->empty());
    int dims = 1;
    for (auto d : inputs[0].dim()) dims *= d;
    TensorShapeDef sdef;
    if (offset_ < 0) {
      CHECK(dims % split_ == 0) << dims << "\t" << split_;
      stride_ = dims / split_;
      offset_ = dims / split_ * index_;
    }
    int count = stride_;
    CHECK(count <= dims);
    sdef.add_dim(count);

    out_shape->push_back(sdef);
    VLOG(V_DEBUG) << out_shape->at(0).DebugString();
    VLOG(V_DEBUG) << op_def_.DebugString();
  }

 private:
  int offset_;
  int stride_;
  int split_;
  int index_;
};

class ConcatOpDecl : public OpDecl {
 public:
  ConcatOpDecl(const OpDef& def) : OpDecl(def) {}
  
  void MakeGradient(vector<OpDef>* grad) override {
    //It needs further design!!!
    CHECK(grad->empty());
    vector<string> input;
    vector<string> input_grad;
    for (auto& s : op_def_.input()) {
      input.push_back(s);
      input_grad.push_back(GetGradientName(s));
    }
    OpDef slice_all;
    OpDefBuilder("SliceAll")
      .Input(GetGradientName(op_def_.output(0)))
      .Input(input)
      .Output(input_grad)
      .Device(op_def_)
      .Finalize(&slice_all);
    grad->push_back(slice_all);
  }

  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(out_shape->empty());
    CHECK(!inputs.empty()); 

    CHECK(inputs[0].dim_size() > 0);
    for (auto& s : inputs) {
      LOG(INFO) << s.DebugString();
      CHECK(s.dim_size() == inputs[0].dim_size());
    }

    int sum = 0;
    for (auto& s : inputs) {
      CHECK(s.dim(0) > 0 || s.dim(0) == -1);
      sum = (s.dim(0) == -1 || sum == -1) ? -1 : (sum + s.dim(0));
      for (int i = 1; i < inputs[0].dim_size(); i++) {
        CHECK(s.dim(i) == inputs[0].dim(i));
      }
    }

    out_shape->resize(1);
    out_shape->at(0) = inputs[0];
    out_shape->at(0).set_dim(0, sum);
    VLOG(V_DEBUG) << out_shape->at(0).DebugString();
  }
};

class SliceAllOpDecl : public OpDecl {
 public:
  SliceAllOpDecl(const OpDef& def) : OpDecl(def) {
    CHECK(def.input_size() >= 2); 
    CHECK(def.output_size() + 1 == def.input_size()); 
  }

  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == op_def_.input_size());
    CHECK(out_shape->empty());
    for (int i = 1; i < inputs.size(); i++) {
      out_shape->push_back(inputs[i]);
    }
  }

};

REGISTER_OP_DECL_BUILDER("Slice" , SliceOpDecl );
REGISTER_OP_DECL_BUILDER("Concat", ConcatOpDecl);
REGISTER_OP_DECL_BUILDER("SliceAll", SliceAllOpDecl);

} //namespace backend
