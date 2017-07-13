#include "cavs/backend/op_decl.h"
#include "cavs/util/op_def_builder.h"

using std::vector;
using std::string;

namespace backend {

class ExtractOpDecl : public OpDecl {
 public:
  ExtractOpDecl(const OpDef& def) : OpDecl(def) {
    CHECK(def.shape_size() == 1)
         << def.DebugString();
  }

  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(out_shape->empty());
    out_shape->resize(1);
    out_shape->at(0).clear_dim();
    out_shape->at(0) = op_def_.shape(0);
  }
};

class GatherOpDecl : public ExtractOpDecl {
 public:
  GatherOpDecl(const OpDef& def) : ExtractOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    //LOG(FATAL) << op_def_.DebugString();
    //for the scatter output, the name is arbitary
    //because the output tensor is set to(share memory with)
    //the message passer tensor during runtime.
    OpDef scatter;
    OpDefBuilder("Scatter")
      .Input(GetGradientName(op_def_.output(0)))
      .Output("Scatter_Backward_"+std::to_string(GetHash(op_def_)))
      .Device(op_def_)
      .Finalize(&scatter);
    grad->push_back(scatter);
  }
};

class FetchUpperGradOpDecl : public ExtractOpDecl {
 public:
  FetchUpperGradOpDecl(const OpDef& def) : ExtractOpDecl(def) {}
};

class PullOpDecl : public ExtractOpDecl {
 public:
  PullOpDecl(const OpDef& def) : ExtractOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    //LOG(FATAL) << op_def_.DebugString();
    OpDef scatter;
    //the shape of pull op is specified during defination,
    //but that of push op is inferenced through shape-inference function.
    //the input of pull operator is the lower layer output(full unit tensor)
    //the output of pull operator is the current layer inner input(partial unit tensor)
    //the input of push operator is the current layer inner error signal(partial unit tensor)
    //the output of push operator is the lower layer error signal(full unit tensor)
    //we do not need to care that push op will put the output to messagepusher,
    //because it is not time-consuming(tensor inner-pointer tranformation).
    OpDefBuilder("Push")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Device(op_def_)
      .Finalize(&scatter);
    grad->push_back(scatter);
  }
};

class EmitOpDecl : public OpDecl {
 public:
  EmitOpDecl(const OpDef& def) : OpDecl(def) {}
  
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(out_shape->empty());
    CHECK(inputs.size() > 0);
    out_shape->resize(1);
    out_shape->at(0) = inputs[0];
  }
};


class ScatterOpDecl : public EmitOpDecl {
 public:
  ScatterOpDecl(const OpDef& def) : EmitOpDecl(def) {}
  void MakeGradient(vector<OpDef>* grad) override {
    //for the frontend, shape item can be omitted
    //but it must be inferenced and set in midend for gradient generation
    CHECK(op_def_.shape_size() == 1) << op_def_.DebugString();
    //LOG(FATAL) << op_def_.DebugString();
    //For tree-lstm, we only scatter the result to one parent,
    //so we need only gather the diff from one parent(offset == 0)
    //gather is a source operator with no inputs,
    //its running input comes from graph scheduler
    OpDef gather;
    OpDefBuilder("Gather")
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .AttrSingle("Child", 0)
      .Device(op_def_)
      .Finalize(&gather);
    grad->push_back(gather);
  }
};

class PushOpDecl : public EmitOpDecl {
 public:
  PushOpDecl(const OpDef& def) : EmitOpDecl(def) {
    //output_name_ = GetSingleArg<string>(def, "OutputName");
    //CHECK(output_name_.length());
  }
  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    EmitOpDecl::ShapeInference(out_shape, inputs);
    CHECK(inputs.size() == 1);
  }
  void MakeGradient(vector<OpDef>* grad) override {
    //LOG(FATAL) << op_def_.DebugString();
    //the shape of pull op is specified during defination,
    //but that of push op is inferenced through shape-inference function.
    //To generate the pull operator, we must assemble the shape iterm.
    //the input of push operator is the current layer inner output(partial unit tensor)
    //the output of push operator is another partial unit tensor(p1)
    //the upper layer full unit tensor is copied from p1 in graphoutput operator
    //the input of pull operator should be the backward error signal of upper layer,
    //its value is set with a starting op(fetchuppergradop)
    //the output of pull operator is the current layer inner error signal(partial unit tensor).
    OpDef pull;
    OpDefBuilder("Pull")
      .Input(GetGradientName(op_def_.output(0)))
      .Output(GetGradientName(op_def_.input(0)))
      .Shape(op_def_)
      .Device(op_def_)
      .Finalize(&pull);
    grad->push_back(pull);
  }
 //private:
  //string output_name_;
};

class GraphOutputOpDecl : public OpDecl {
 public:
  GraphOutputOpDecl(const OpDef& def) : OpDecl(def) {
    CHECK(def.shape_size() == 1)
         << def.DebugString();
  }

  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(out_shape->empty());
    out_shape->push_back(op_def_.shape(0));
  }

  void MakeGradient(vector<OpDef>* grad) override {
    CHECK(op_def_.input_size() >= 2) << op_def_.DebugString();
    CHECK(op_def_.output_size() == 1) << op_def_.DebugString();
    vector<string> var;
    vector<string> var_grad;
    for (int i = 2; i < op_def_.input_size(); i++) {
      var.push_back(op_def_.input(i));
      var_grad.push_back(GetGradientName(op_def_.input(i)));
    }
    OpDef graphout_grad;
    OpDefBuilder(GetGradientName("GraphOutput"))
      //.Input(op_def_.input(0))
      //.Input(op_def_.input(1))
      .Input(GetGradientName(op_def_.output(0)))
      .Input(var)
      //Output should be the gradient of the variable
      .Output(var_grad)
      .Device(op_def_)
      .Finalize(&graphout_grad);
    grad->push_back(graphout_grad);
  }
};

class GraphOutputGradOpDecl : public EmitOpDecl {
 public:
  GraphOutputGradOpDecl(const OpDef& def) : EmitOpDecl(def) {}

  void ShapeInference(vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) override {
    CHECK(out_shape->empty());
    CHECK(inputs.size() > 1);
    for (int i = 1; i < inputs.size(); i++) {
      out_shape->push_back(inputs[i]);
    }
  }
};

REGISTER_OP_DECL_BUILDER("Gather",         GatherOpDecl        );
REGISTER_OP_DECL_BUILDER("Pull",           PullOpDecl          );
REGISTER_OP_DECL_BUILDER("Scatter",        ScatterOpDecl       );
REGISTER_OP_DECL_BUILDER("Push",           PushOpDecl          );
REGISTER_OP_DECL_BUILDER("FetchUpperGrad", FetchUpperGradOpDecl);
REGISTER_OP_DECL_BUILDER("GraphOutput",    GraphOutputOpDecl   );
REGISTER_OP_DECL_BUILDER(GetGradientName("GraphOutput"), GraphOutputGradOpDecl);

} //namespace backend
