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
    OpDef assign_def;
    OpDefBuilder("Assign")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .Device(op_def_)
      .Finalize(&assign_def);
    grad->push_back(std::move(assign_def));
  }
};

class AddOpDecl : public BinaryOpDecl {
 public:
  explicit AddOpDecl (const OpDef& def) : BinaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->empty());
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    for (int i = 0; i < op_def_.input_size(); i++) {
      if (partial_ != i) {
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
  }
};

class SubOpDecl : public BinaryOpDecl {
 public:
  SubOpDecl(const OpDef& def) : BinaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->empty());
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    if (partial_ != 0) {
      OpDef assign_def;
      OpDefBuilder("Assign")
        .Input(GetGradientName(op_def_.output(0)))
        .Output(GetGradientName(op_def_.input(0)))
        .Shape(op_def_)
        .Device(op_def_)
        .Finalize(&assign_def);
      grad->push_back(std::move(assign_def));
    }
    if (partial_ != 1) {
      OpDef neg_def;
      OpDefBuilder("Neg")
        .Input(GetGradientName(op_def_.output(0)))
        .Output(GetGradientName(op_def_.input(1)))
        .Shape(op_def_)
        .Device(op_def_)
        .Finalize(&neg_def);
      grad->push_back(std::move(neg_def));
    }
  }
};

class MulOpDecl : public BinaryOpDecl {
 public:
  MulOpDecl(const OpDef& def) : BinaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->empty());
    CHECK(op_def_.input_size() == 2);
    CHECK(op_def_.output_size() == 1);
    if (partial_ != 0) {
      OpDef mul_def_0;
      OpDefBuilder("Mul")
        .Input(GetGradientName(op_def_.output(0)))
        .Input(op_def_.input(1))
        .Output(GetGradientName(op_def_.input(0)))
        //.Shape(op_def_)
        .Device(op_def_)
        .Finalize(&mul_def_0);
      grad->push_back(std::move(mul_def_0));
    }
    if (partial_ != 1) {
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
  }
};

//partial mul can replace the scal op for the front-end operations
class ScalOpDecl : public UnaryOpDecl {
 public:
  ScalOpDecl(const OpDef& def) : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->empty());
    float alpha = GetSingleArg<float>(op_def_, "alpha");
    OpDef scal_def;
    OpDefBuilder("Scal")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .AttrSingle<float>("alpha", alpha)
      .Device(op_def_)
      .Finalize(&scal_def);
    grad->push_back(std::move(scal_def));
  }
};

class NegOpDecl : public UnaryOpDecl {
 public:
  NegOpDecl(const OpDef& def) : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->empty());
    OpDef scal_def;
    OpDefBuilder("Scal")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .AttrSingle<float>("alpha", -1.f)
      .Device(op_def_)
      .Finalize(&scal_def);
    grad->push_back(std::move(scal_def));
  }
};


class FillOpDecl : public OpDecl {
 public:
  explicit FillOpDecl(const OpDef& def) :
    OpDecl(def) {}
  void ShapeInference(std::vector<TensorShapeDef>* out_shape,
    const std::vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 2) << op_def_.DebugString();
    CHECK(op_def_.shape_size() == 0);
    CHECK(inputs[0].dim_size() == 1);
    CHECK(inputs[0].dim(0) == 1);
    CHECK(inputs[1].dim_size() > 0);
    CHECK(out_shape->empty());
    out_shape->push_back(inputs[1]);
  }
  //no gradient needed
};

class SquareOpDecl : public UnaryOpDecl {
 public:
  explicit SquareOpDecl(const OpDef& def) : UnaryOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(grad->empty());
    CHECK(op_def_.input_size() == 1);
    CHECK(op_def_.output_size() == 1);
    OpDef mul_def;
    OpDefBuilder(GetGradientName("Square"))
      .Input(op_def_.input(0))
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .AttrSingle<float>("alpha", 2.f)
      .Device(op_def_)
      .Finalize(&mul_def);
    grad->push_back(std::move(mul_def));
  }
};

class SquareGradOpDecl : public BinaryOpDecl {
 public:
  explicit SquareGradOpDecl(const OpDef& def) : BinaryOpDecl(def) {}
  //no gradient
};

REGISTER_OP_DECL_BUILDER("Assign", AssignOpDecl);
REGISTER_OP_DECL_BUILDER("Add", AddOpDecl);
REGISTER_OP_DECL_BUILDER("Sub", SubOpDecl);
REGISTER_OP_DECL_BUILDER("Mul", MulOpDecl);
REGISTER_OP_DECL_BUILDER("Scal", ScalOpDecl);
REGISTER_OP_DECL_BUILDER("Neg", NegOpDecl);
REGISTER_OP_DECL_BUILDER("Square", SquareOpDecl);
REGISTER_OP_DECL_BUILDER(GetGradientName("Square"), SquareGradOpDecl);
REGISTER_OP_DECL_BUILDER("Fill", FillOpDecl);

} //namespace backend
