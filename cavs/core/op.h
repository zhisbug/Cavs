#ifndef CAVS_CORE_OP_H_
#define CAVS_CORE_OP_H_

#include "cavs/core/op_def.pb.h"
#include "cavs/core/tensor.h"
#include "cavs/core/session.h"

namespace cavs {

class Op {
 public:
  explicit Op(const OpDef& def, Session *s); 
  virtual void Compute() = 0;
  inline const Tensor* Input(int idx) const { return inputs_.at(idx); }
  inline Tensor* Output(int idx) { return outputs_.at(idx); }
 private:
  vector<const Tensor*> inputs_;
  vector<Tensor*> outputs_;
};

Op* CreateOp(const OpDef& def, Session *s);

#define REGISTER_OP_BUILDER(key, ...)                       \
    REGISTER_OP_BUILDER_UNIQ(__COUNTER__, key, __VA_ARGS__)
#define REGISTER_OP_BUILDER_UNIQ(ctr, key, ...)             \
    REGISTER_OP_BUILDER_CONCAT(ctr, key, __VA_ARGS__)
#define REGISTER_OP_BUILDER_CONCAT(ctr, key, ...)           \
    static op_factory::OpRegister register_body_##ctr##_op( \
        op_factory::key.LowerToString(),                    \
            [](const OpDef& def, Session* s) -> Op* {       \
                    return new __VA_ARGS__(def, s);         \
                })

namespace op_factory {

class Key {
 public:
  Key(const string& name) : op_name_(name), dev_(""), label_("") {}
  Key(const OpDef& def) 
      : op_name_(def.name()), label_(def.label()) {
          if (def.device() == OpDef::GPU)
              dev_ = "GPU";
          else
              dev_ = "CPU";
    }
  Key& Device(string dev) { dev_ = dev; return *this; }
  Key& Label(string label) { label_ = label; return *this; }
  string LowerToString() const { return op_name_+":"+dev_+":"+label_; }
 private:
  string op_name_;
  string dev_;
  string label_;
};

class OpRegister {
 public:
  typedef Op* (*Factory)(const OpDef& def, Session* s);

  OpRegister(const string& name, Factory factory) {
    InitInternal(name, factory); 
  }
 private:
  void InitInternal(const string& name, Factory factory); 
};

} //namespace op_factory

} //namespace cavs
        
#endif
