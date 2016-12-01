#ifndef CAVS_MIDEND_SESSION_H_
#define CAVS_MIDEND_SESSION_H_

#include "cavs/midend/op_chain_def.pb.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/op.h"

namespace cavs {

class SessionBase {
 public:
  SessionBase() {}
  SessionBase(const OpChainDef& def) : op_chain_def_(def) {}
  const Tensor* GetTensor(const string& name) const;
  void InsertTensor(const Tensor* t);
  virtual void Run() = 0;
 protected:
  //Tensor* CreateTensor(const OpDef& op_def);
  unordered_map<string, const Tensor*> tensor_map_;
  OpChainDef op_chain_def_;
};

class Op;
class OpContext;
class SimpleSession : public SessionBase {
 public:
  SimpleSession(const OpChainDef& def);
  void Run() override;
 private:
  std::vector<std::pair<Op*, OpContext*>> executors_;
};

} //namespace cavs

#endif
