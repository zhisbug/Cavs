#include "cavs/backend/op_decl_elementwise.h"

using std::vector;

namespace backend {

class AssignOpDecl : public UnaryOpDecl {
 public:
  explicit AssignOpDecl(const OpDef& def) : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 1);
    CHECK(op_def_.output_size() == 1);
    for (int i = 0; i < op_def_.input_size(); i++) {
      OpDef assign_def;
      OpDefBuilder("Assign")
        .Input(GetGradientName(op_def_.output(0)))
        .Output(GetGradientName(op_def_.input(0)))
        .Shape(op_def_)
        .Device(op_def_)
        .Finalize(&assign_def);
      grad->push_back(std::move(assign_def));
    }
  }
};

class AddOpDecl : public BinaryOpDecl {
 public:
  explicit AddOpDecl (const OpDef& def) : BinaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    for (int i = 0; i < op_def_.input_size(); i++) {
      OpDef assign_def;
      OpDefBuilder("Assign")
        .Input(GetGradientName(op_def_.output(0)))
        .Output(GetGradientName(op_def_.input(i)))
        .Shape(op_def_)
        .Device(op_def_)
        .Finalize(&assign_def);
      grad->push_back(std::move(assign_def));
    }
  }
};

class SubOpDecl : public BinaryOpDecl {
 public:
  SubOpDecl(const OpDef& def) : BinaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    OpDef assign_def;
    OpDefBuilder("Assign")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .Device(op_def_)
      .Finalize(&assign_def);
    grad->push_back(std::move(assign_def));
    OpDef neg_def;
    OpDefBuilder("Neg")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(1)))
      .Shape(op_def_)
      .Device(op_def_)
      .Finalize(&neg_def);
    grad->push_back(std::move(neg_def));
  }
};

class MulOpDecl : public BinaryOpDecl {
 public:
  MulOpDecl(const OpDef& def) : BinaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    OpDef mul_def_0;
    OpDefBuilder("Mul")
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(1))
      .Output(GetGradientName(op_def_.input(0)))
      //.Shape(op_def_)
      .Device(op_def_)
      .Finalize(&mul_def_0);
    grad->push_back(std::move(mul_def_0));
    OpDef mul_def_1;
    OpDefBuilder("Mul")
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(0))
      .Output(GetGradientName(op_def_.input(1)))
      //.Shape(op_def_)
      .Device(op_def_)
      .Finalize(&mul_def_1);
    grad->push_back(std::move(mul_def_1));
  }
};

class ScalOpDecl : public MulOpDecl {
 public:
  ScalOpDecl(const OpDef& def) : MulOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 1);
    CHECK(op_def_.output_size() == 1);
    OpDef right_def;
    BuildConstantOpDef(&right_def, 
        GetGradientName(op_def_.input(0)),
        op_def_.shape(0),
        GetConstFromConstantOp(op_def_));
    grad->push_back(std::move(right_def));
  }
};

class NegOpDecl : public UnaryOpDecl {
 public:
  NegOpDecl(const OpDef& def) : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    grad->clear();
    CHECK(op_def_.input_size() == 1);
    CHECK(op_def_.output_size() == 1);
    OpDef right_def;
    BuildConstantOpDef(&right_def, 
        GetGradientName(op_def_.input(0)),
        op_def_.shape(0),
        -1.f);
    grad->push_back(std::move(right_def));
  }
};

REGISTER_OP_DECL_BUILDER("Assign", AssignOpDecl);
REGISTER_OP_DECL_BUILDER("Add", AddOpDecl);
REGISTER_OP_DECL_BUILDER("Sub", SubOpDecl);
REGISTER_OP_DECL_BUILDER("Mul", MulOpDecl);
REGISTER_OP_DECL_BUILDER("Scal", ScalOpDecl);
REGISTER_OP_DECL_BUILDER("Neg", NegOpDecl);

} //namespace backend
