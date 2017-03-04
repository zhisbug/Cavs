#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"

using std::vector;

namespace backend {

class MatMulOpDecl : public OpDecl {
 public:
  MatMulOpDecl(const OpDef& def) : OpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    OpDef mul_def_0;
    OpDefBuilder("MatMul")
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(1))
      .Output(GetGradientName(op_def_.input(0)))
      .Device(op_def_)
      //.AttrList("Transpose", 1)
      .AttrList("Transpose", TransMask({{1,true}}))
      .Finalize(&mul_def_0);
    grad->push_back(std::move(mul_def_0));
    OpDef mul_def_1;
    OpDefBuilder("MatMul")
      .Input(op_def_.input(0))
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(1)))
      .Device(op_def_)
      //.AttrList("Transpose", 0)
      .AttrList("Transpose", TransMask({{0, true}}))
      .Finalize(&mul_def_1);
    grad->push_back(std::move(mul_def_1));
  }
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 2) << inputs.size();
    CHECK(inputs[0].dim_size() == 2);
    CHECK(inputs[1].dim_size() == 2);
    bool TransA = false;
    bool TransB = false;
    //LOG(INFO) << op_def_.DebugString();
    for (auto& t : GetListArg<int>(op_def_, "Transpose")) {
      if (t == 0) TransA = true;
      if (t == 1) TransB = true;
    }
    int MA = (TransA == false)? inputs[0].dim(0) : inputs[0].dim(1);
    int KA = (TransA == false)? inputs[0].dim(1) : inputs[0].dim(0);
    int KB = (TransB == false)? inputs[1].dim(0) : inputs[1].dim(1);
    int NB = (TransB == false)? inputs[1].dim(1) : inputs[1].dim(0);
    CHECK(KA == KB) << "KA: " << KA << "\tKB: " << KB
                    << op_def_.DebugString();
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0).add_dim(MA);
    out_shape->at(0).add_dim(NB);
  }

 private:
  vector<int> TransMask(const vector<std::pair<int, bool>>& trans) {
    vector<int> input = GetListArg<int>(op_def_, "Transpose");
    vector<int> output;
    for (auto& one_pair : trans) {
      bool trans_input =
        std::find(input.begin(), input.end(), one_pair.first) != input.end();
      if (trans_input ^ one_pair.second)
        output.push_back(one_pair.first);
    }
    return output;
  }
};

REGISTER_OP_DECL_BUILDER("MatMul", MatMulOpDecl);

}; //namespace backend
