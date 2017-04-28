#ifndef CAVS_BACKEND_OP_IMPL_H_
#define CAVS_BACKEND_OP_IMPL_H_

#include "cavs/midend/op_context.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/op_util.h"

#include <vector>

namespace backend {

using ::midend::OpContext;

class OpImpl {
 public:
  explicit OpImpl(const OpDef& def) : op_def_(def) {}
  //explicit Op(const OpDef& def): name_(def.name()) {}
  virtual void Compute(OpContext* context) = 0;
  std::string DebugInfo(int level=V_DEBUG) const {
    if (level == V_DEBUG)
      return op_def_.DebugString(); 
    else
      return op_def_.name();
  }
 protected:
  OpDef op_def_;
};

OpImpl* CreateOp(const OpDef& def);

#define REGISTER_OP_IMPL_BUILDER(key, ...)                         \
    REGISTER_OP_IMPL_BUILDER_UNIQ(__COUNTER__, key, __VA_ARGS__)
#define REGISTER_OP_IMPL_BUILDER_UNIQ(ctr, key, ...)               \
    REGISTER_OP_IMPL_BUILDER_CONCAT(ctr, key, __VA_ARGS__)
#define REGISTER_OP_IMPL_BUILDER_CONCAT(ctr, key, ...)             \
    static op_factory::OpImplRegister                              \
      register_body_##ctr##_op_impl(                               \
        op_factory::key.ToString(),                                \
          [](const OpDef& def) -> OpImpl* {                        \
              return new __VA_ARGS__(def);                         \
            });

namespace op_factory {

class Key {
 public:
  Key(const std::string& name) 
      : op_name_(name), dev_(""), label_("") {}
  Key(const OpDef& def) 
      : op_name_(def.name()), label_(def.label()) {
    dev_ = ::midend::DeviceTypeToString(def.device());
  }
  Key& Device(std::string dev) { dev_ = dev; return *this; }
  Key& Label(std::string label) { label_ = label; return *this; }
  std::string ToString() const { return op_name_+":"+dev_+":"+label_; }

 private:
  std::string op_name_;
  std::string dev_;
  std::string label_;
};

class OpImplRegister {
 public:
  typedef OpImpl* (*Factory)(const OpDef& def);

  OpImplRegister(const std::string& name, Factory factory) {
    InitInternal(name, factory); 
  }

 private:
  void InitInternal(const std::string& name, Factory factory); 
};

} //namespace op_factory

} //namespace backend 
        
#endif
