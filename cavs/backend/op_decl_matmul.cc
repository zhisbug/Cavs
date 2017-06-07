#include "cavs/backend/op_decl.h"
#include "cavs/util/op_def_builder.h"

using std::vector;

namespace backend {

class MatMulOpDecl : public OpDecl {
 public:
  MatMulOpDecl(const OpDef& def)
    : OpDecl(def), TransA_(false), TransB_(false) {
    for (auto& t : GetListArg<int>(op_def_, "Transpose")) {
      if (t == 0) TransA_ = true;
      else if (t == 1) TransB_ = true;
      else LOG(FATAL) << "Invalid transpose idx: " << t;
    }
  }
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    OpDef mul_def_0;
    if (!TransA_) {
      vector<int> t0 = {1};
      vector<int> t1 = {};
      OpDefBuilder("MatMul")
        .Input(GetGradientName(op_def_.output(0)))
        .Input(op_def_.input(1))
        .Output(GetGradientName(op_def_.input(0)))
        .Device(op_def_)
        .AttrList("Transpose", (TransB_^true ? t0 : t1))
        .Finalize(&mul_def_0);
    }else {
      vector<int> t0 = {0, 1};
      vector<int> t1 = {1};
      OpDefBuilder("MatMul")
        .Input(op_def_.input(1))
        .Input(GetGradientName(op_def_.output(0)))
        .Output(GetGradientName(op_def_.input(0)))
        .Device(op_def_)
        .AttrList("Transpose", TransB_^false ? t0 : t1)
        .Finalize(&mul_def_0);
    }
    grad->push_back(std::move(mul_def_0));
    OpDef mul_def_1;

    if (!TransB_) {
      vector<int> t0 = {0};
      vector<int> t1 = {};
      OpDefBuilder("MatMul")
        .Input(op_def_.input(0))
        .Input(GetGradientName(op_def_.output(0)))
        .Output(GetGradientName(op_def_.input(1)))
        .Device(op_def_)
        .AttrList("Transpose", TransA_^true ? t0 : t1)
        .Finalize(&mul_def_1);
    }else {
      vector<int> t0 = {0, 1};
      vector<int> t1 = {0};
      OpDefBuilder("MatMul")
        .Input(GetGradientName(op_def_.output(0)))
        .Input(op_def_.input(0))
        .Output(GetGradientName(op_def_.input(1)))
        .Device(op_def_)
        .AttrList("Transpose", TransA_^false? t0 : t1)
        .Finalize(&mul_def_1);
    }
    grad->push_back(std::move(mul_def_1));
  }
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() >= 2) << inputs.size();
    CHECK(inputs[0].dim_size() == 2);
    CHECK(inputs[1].dim_size() == 2);
    int MA = (TransA_ == false)? inputs[0].dim(0) : inputs[0].dim(1);
    int KA = (TransA_ == false)? inputs[0].dim(1) : inputs[0].dim(0);
    int KB = (TransB_ == false)? inputs[1].dim(0) : inputs[1].dim(1);
    int NB = (TransB_ == false)? inputs[1].dim(1) : inputs[1].dim(0);
    CHECK(KA == KB) << "KA: " << KA << "\tKB: " << KB
                    << op_def_.DebugString();
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0).add_dim(MA);
    out_shape->at(0).add_dim(NB);
  }

 private:
  bool TransA_;
  bool TransB_;
};

REGISTER_OP_DECL_BUILDER("MatMul", MatMulOpDecl);

}; //namespace backend
