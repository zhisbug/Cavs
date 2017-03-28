#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"
#include "cavs/util/op_util.h"

namespace backend {

class UnaryOpDecl : public OpDecl {
 public:
  explicit UnaryOpDecl(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(std::vector<TensorShapeDef>* out_shape,
    const std::vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 1) << op_def_.DebugString();
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0) = inputs[0];
  }

 protected:
  void MakeGradientUnary(std::vector<OpDef>* grad) {
    CHECK(grad->size() == 0);
    CHECK(op_def_.input_size() == 1);
    CHECK(op_def_.output_size() == 1);
    OpDef grad_def;
    OpDefBuilder(GetGradientName(op_def_.name()))
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .Device(op_def_)
      .Finalize(&grad_def);
    grad->push_back(std::move(grad_def));
  }
};

class BinaryOpDecl : public OpDecl {
 public:
  explicit BinaryOpDecl(const OpDef& def) :
    OpDecl(def), partial_(-1) {}
  void ShapeInference(std::vector<TensorShapeDef>* out_shape,
    const std::vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 2) << inputs.size();
    CHECK(out_shape && out_shape->empty());
    if (inputs[0].dim_size() == inputs[1].dim_size()) {
      for (unsigned i = 0; i < inputs[0].dim_size(); i++)
        CHECK(inputs[0].dim(i) == inputs[1].dim(i)) 
          << inputs[0].DebugString() << inputs[1].DebugString()
          << op_def_.DebugString();
      out_shape->push_back(inputs[0]);
    }else if (inputs[0].dim_size() == 1 && inputs[0].dim(0) == 1) {
      out_shape->push_back(inputs[1]);
      partial_ = 0;
    }else if (inputs[1].dim_size() == 1 && inputs[1].dim(0) == 1) {
      out_shape->push_back(inputs[0]);
      partial_ = 1;
    }else {
      LOG(FATAL) << inputs[0].DebugString() << "\n"
                 << inputs[1].DebugString() << "\n"
                 << op_def_.DebugString();
    }
  }

 protected:
  int partial_;
};

class TernaryOpDecl : public OpDecl {
 public:
  explicit TernaryOpDecl(const OpDef& def) : OpDecl(def) {}
  void ShapeInference(std::vector<TensorShapeDef>* out_shape,
    const std::vector<TensorShapeDef>& inputs) override {
    CHECK(inputs.size() == 3) << inputs.size();
    CHECK(inputs[0].dim_size() == inputs[1].dim_size());
    CHECK(inputs[1].dim_size() == inputs[2].dim_size());
    for (unsigned i = 0; i < inputs[0].dim_size(); i++) {
      CHECK(inputs[0].dim(i) == inputs[1].dim(i));
      CHECK(inputs[1].dim(i) == inputs[2].dim(i));
    }
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0) = inputs[0];
  }
};

} //namespace backend
