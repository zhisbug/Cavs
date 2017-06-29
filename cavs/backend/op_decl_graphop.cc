#include "cavs/backend/op_decl.h"
#include "cavs/util/op_def_builder.h"

using std::vector;
using std::string;

namespace backend {

class ExtractOpDecl : public OpDecl {
 public:
  ExtractOpDecl(const OpDef& def) : OpDecl(def) {
    CHECK(def.shape_size() == 1);  
  }
  
  void MakeGradient(vector<OpDef>* grad) override {
    LOG(FATAL) << "Not implemented yet";
  }

  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(out_shape->empty());
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0) = op_def_.shape(0);
  }
};

class EmitOpDecl : public OpDecl {
 public:
  EmitOpDecl(const OpDef& def) : OpDecl(def) {}
  
  void MakeGradient(vector<OpDef>* grad) override {
    LOG(FATAL) << "Not implemented yet";
  }

  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(out_shape->empty());
    CHECK(inputs.size() > 0);
    out_shape->resize(1);
    out_shape->at(0) = inputs[0];
  }
};

class GatherOpDecl : public ExtractOpDecl {
 public:
  GatherOpDecl(const OpDef& def) : ExtractOpDecl(def) {}
};

class PullOpDecl : public ExtractOpDecl {
 public:
  PullOpDecl(const OpDef& def) : ExtractOpDecl(def) {}
};

class ScatterOpDecl : public EmitOpDecl {
 public:
  ScatterOpDecl(const OpDef& def) : EmitOpDecl(def) {}
};

class PushOpDecl : public EmitOpDecl {
 public:
  PushOpDecl(const OpDef& def) : EmitOpDecl(def) {}
};

class GraphOutputOpDecl : public EmitOpDecl {
 public:
  GraphOutputOpDecl(const OpDef& def) : EmitOpDecl(def) {}

  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(op_def_.input_size() >= 2) << op_def_.DebugString();
    CHECK(op_def_.output_size() == 1) << op_def_.DebugString();
    vector<string> var_grad;
    for (int i = 2; i < op_def_.input_size(); i++)
      var_grad.push_back(GetGradientName(op_def_.input(i)));
    OpDef graphout_grad;
    OpDefBuilder(GetGradientName("GraphOutput"))
      .Input(GetGradientName(op_def_.output(0)))
      .Input(op_def_.input(0))
      .Input(op_def_.input(1))
      //Output should be the gradient of the variable
      .Output(var_grad)
      .Device(op_def_)
      .Finalize(&graphout_grad);
    grad->push_back(std::move(graphout_grad));
  }
};

REGISTER_OP_DECL_BUILDER("Gather",      GatherOpDecl     );
REGISTER_OP_DECL_BUILDER("Pull",        PullOpDecl       );
REGISTER_OP_DECL_BUILDER("Scatter",     ScatterOpDecl    );
REGISTER_OP_DECL_BUILDER("Push",        PushOpDecl       );
REGISTER_OP_DECL_BUILDER("GraphOutput", GraphOutputOpDecl);

} //namespace backend
