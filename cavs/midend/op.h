#ifndef CAVS_MIDEND_OP_H_
#define CAVS_MIDEND_OP_H_

#include "cavs/midend/op_def.pb.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/session.h"

namespace midend {

class OpContext {
 public:
  OpContext(const OpDef& op_def, SessionBase* sb);
  inline const Tensor& Input(int idx) { return *(inputs_.at(idx)); }
  inline Tensor* Output(int idx) { return outputs_.at(idx); }
 private:
  vector<const Tensor*> inputs_;
  vector<Tensor*> outputs_;
};

class Op {
 public:
  explicit Op(const OpDef& def): name_(def.name()) {}
  FORCE_INLINE const string& name() const { return name_; }
  virtual void Compute(OpContext* context) = 0;
 private:
  string name_;
};


#define REGISTER_OP_BUILDER(key, ...)                       \
    REGISTER_OP_BUILDER_UNIQ(__COUNTER__, key, __VA_ARGS__)
#define REGISTER_OP_BUILDER_UNIQ(ctr, key, ...)             \
    REGISTER_OP_BUILDER_CONCAT(ctr, key, __VA_ARGS__)
#define REGISTER_OP_BUILDER_CONCAT(ctr, key, ...)           \
    static ::midend::op_factory::OpRegister                 \
      register_body_##ctr##_op(                             \
        ::midend::op_factory::key.ToString(),               \
          [](const ::midend::OpDef& def) -> ::midend::Op* { \
              return new __VA_ARGS__(def);                  \
            })


Op* CreateOp(const OpDef& def);

namespace op_factory {

class Key {
 public:
  Key(const string& name) 
      : op_name_(name), dev_(""), label_("") {}
  Key(const OpDef& def) 
      : op_name_(def.name()), label_(def.label()) {
    //if (def.device() == GPU)
      //dev_ = "GPU";
    //else
      //dev_ = "CPU";
    dev_ = DeviceTypeToString(def.device());
  }
  Key& Device(string dev) { dev_ = dev; return *this; }
  Key& Label(string label) { label_ = label; return *this; }
  string ToString() const { return op_name_+":"+dev_+":"+label_; }
 private:
  string op_name_;
  string dev_;
  string label_;
};

class OpRegister {
 public:
  typedef Op* (*Factory)(const OpDef& def);

  OpRegister(const string& name, Factory factory) {
    InitInternal(name, factory); 
  }
 private:
  void InitInternal(const string& name, Factory factory); 
};

} //namespace op_factory

} //namespace midend
        
#endif
