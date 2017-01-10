#ifndef CAVS_BACKEND_OP_DECL_H_
#define CAVS_BACKEND_OP_DECL_H_

#include "cavs/midend/tensor.h"
#include "cavs/midend/session.h"
#include "cavs/midend/op_context.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/proto/tensor_shape.pb.h"

namespace backend {

class OpDecl {
 public:
  explicit OpDecl(const OpDef& def) : op_def_(def) {};
  virtual void MakeGradient(vector<OpDef>* grad) = 0;
  virtual void ShapeInference(
    vector<TensorShapeDef>* out_shape,
    const vector<TensorShapeDef>& inputs) = 0;

 protected:
  OpDef op_def_;
  inline std::string GetGradientName(const std::string& op) {
    return op+"_grad";
  }
};

vector<OpDef> MakeGradient(const OpDef& def);
vector<TensorShapeDef> ShapeInference(const OpDef& def, 
    const vector<TensorShapeDef>& inputs);

#define REGISTER_OP_DECL_BUILDER(key, ...)                         \
    REGISTER_OP_DECL_BUILDER_UNIQ(__COUNTER__, key, __VA_ARGS__)
#define REGISTER_OP_DECL_BUILDER_UNIQ(ctr, key, ...)               \
    REGISTER_OP_DECL_BUILDER_CONCAT(ctr, key, __VA_ARGS__)
#define REGISTER_OP_DECL_BUILDER_CONCAT(ctr, key, ...)             \
    static op_factory::OpDeclRegister                              \
      register_body_##ctr##_op_decl(key,                           \
          [](const OpDef& def) -> OpDecl* {                        \
              return new __VA_ARGS__(def);                         \
            });                                                    

namespace op_factory {

class OpDeclRegister {
 public:
  typedef OpDecl* (*Factory)(const OpDef& def);

  OpDeclRegister(const string& name, Factory factory) {
    InitInternal(name, factory); 
  }

 private:
  void InitInternal(const string& name, Factory factory); 
};

} //namespace op_factory

} //namespace backend 
        
#endif
