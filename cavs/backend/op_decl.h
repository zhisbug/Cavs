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
  virtual void MakeGradient(std::vector<OpDef>* grad) = 0;
  virtual void ShapeInference(
    std::vector<TensorShapeDef>* out_shape,
    const std::vector<TensorShapeDef>& inputs) = 0;
  inline static std::string GetGradientName(const std::string& op) {
    return op+"_grad";
  }
  inline static std::string GetOriginName(const std::string& op) {
    CHECK(op.length() > 5);
    CHECK(op.substr(op.length()-5, 5) == "_grad");
    return op.substr(0, op.length()-5);
  }

 protected:
  OpDef op_def_;
};

std::vector<OpDef> MakeGradient(const OpDef& def);
std::vector<TensorShapeDef> ShapeInference(const OpDef& def, 
    const std::vector<TensorShapeDef>& inputs);

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

  OpDeclRegister(const std::string& name, Factory factory) {
    InitInternal(name, factory); 
  }

 private:
  void InitInternal(const std::string& name, Factory factory); 
};

} //namespace op_factory

} //namespace backend 
        
#endif
