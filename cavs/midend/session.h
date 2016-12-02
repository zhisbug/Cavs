#ifndef CAVS_MIDEND_SESSION_H_
#define CAVS_MIDEND_SESSION_H_

#include "cavs/midend/op_chain_def.pb.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/op.h"

namespace cavs {

class SessionBase {
 public:
  SessionBase() {}
  //SessionBase(const OpChainDef& def) : op_chain_def_(def) {}
  virtual void SetOpChainDef(const OpChainDef& def) {
    op_chain_def_ = def;
  }
  const Tensor* GetTensor(const string& name) const;
  void InsertTensor(const Tensor* t);
  virtual void Run() = 0;
 protected:
  //Tensor* CreateTensor(const OpDef& op_def);
  unordered_map<string, const Tensor*> tensor_map_;
  OpChainDef op_chain_def_;
};

SessionBase* GetSession(const string& name);

#define REGISTER_SESSION_BUILDER(key, sess)                  \
    REGISTER_SESSION_BUILDER_UNIQ(__COUNTER__, key, sess)
#define REGISTER_SESSION_BUILDER_UNIQ(ctr, key, sess)       \
    REGISTER_SESSION_BUILDER_CONCAT(ctr, key, sess)
#define REGISTER_SESSION_BUILDER_CONCAT(ctr, key, sess)     \
    static session_factory::SessionRegister                \
        register_body_##ctr##_session(key, sess)

namespace session_factory {

class SessionRegister {
 public:
  SessionRegister(const string& name, Session* sess) {
    InitInternal(name, sess); 
  }
 private:
  void InitInternal(const string& name, Session* sess); 
};

} //namespace session_factory

} //namespace cavs

#endif
