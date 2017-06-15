#include "cavs/backend/op_decl.h"
#include "cavs/util/op_def_builder.h"

using std::vector;

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
};

REGISTER_OP_DECL_BUILDER("Gather",      GatherOpDecl     );
REGISTER_OP_DECL_BUILDER("Pull",        PullOpDecl       );
REGISTER_OP_DECL_BUILDER("Scatter",     ScatterOpDecl    );
REGISTER_OP_DECL_BUILDER("Push",        PushOpDecl       );
REGISTER_OP_DECL_BUILDER("GraphOutput", GraphOutputOpDecl);

} //namespace backend
