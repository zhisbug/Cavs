#ifndef CAVS_CORE_OP_H_
#define CAVS_CORE_OP_H_

#include "cavs/core/op_def.pb.h"
#include "cavs/core/tensor.h"
#include "cavs/core/session.h"

namespace cavs {

class Op {
 public:
  explicit Op(const OpDef& def, Session *s); 
  //explicit Op(const vector<OpDef>& def, Session *s); //for JIT
  virtual void Compute() = 0;
  inline const Tensor* Input(int idx) const { return inputs_.at(idx); }
  inline Tensor* Outnput(int idx) { return outputs_.at(idx); }
 private:
  vector<const Tensor*> inputs_;
  vector<Tensor*> outputs_;
};

#define REGISTER_OP_BUILDER(key, constructor)           \
    static OpRegister register__body__##name##(         \
        op_factory::key.LowerToString(),                \
            [](OpDef& def) -> Op* {                     \
                    return new constructor(def);        \
                });

namespace op_factory {

class Key {
 public:
  Key(string name) : op_name_(name), dev_(""), label_("") {}
  Key& Device(string dev) { dev_ = dev; }
  Key& Label(string label) { label_ = label; }
  string LowerToString() { return op_name_+":"+dev_+":"+label_; }
 private:
  string op_name_;
  string dev_;
  string label_;
};

class OpRegister {
 public:
  typedef Op* (*Factory)(OpDef& def);

  OpRegister(const string& name, Factory factory) {
    InitInternal(name, factory); 
  }
 private:
  void InitInternal(const string& name, Factory factory); 
};

} //namespace op_factory

} //namespace cavs
        
#endif
