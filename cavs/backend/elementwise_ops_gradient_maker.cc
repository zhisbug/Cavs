#include "cavs/midend/op_gradient_maker.h"

namespace backend {

using ::midend::OpGradientMaker;
using ::midend::OpDef;

class AddOpGradientMaker : public OpGradientMaker {
 public:
  AddOpGradientMaker(const OpDef& def) :
    OpGradientMaker(def) {}
  void MakeGradient(vector<OpDef>* def) override {
    def->clear();
  }
};

class SubOpGradientMaker : public OpGradientMaker {
 public:
  SubOpGradientMaker(const OpDef& def) :
    OpGradientMaker(def) {}
  void MakeGradient(vector<OpDef>* def) override {
    OpDef grad_def = op_def_; 
    grad_def.set_name("Neg");
    grad_def.clear_input();
    for (auto& out : op_def_.output())
      grad_def.add_input(out);
    grad_def_.clear_output();
    for (auto& inp : this->op_def_.input())
      grad_def_add_output(inp);
  }
};


} //namespace backend
