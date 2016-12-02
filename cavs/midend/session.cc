#include "cavs/midend/session.h"
#include "cavs/midend/allocator.h"
#include "cavs/util/logging.h"

namespace cavs {

const Tensor* SessionBase::GetTensor(const string& name) const {
  if (tensor_map_.count(name) == 0)
    return NULL;
  else
    return tensor_map_.at(name);
}

void SessionBase::InsertTensor(const Tensor* t){
  CHECK(tensor_map_.count(t->name()) == 0);
  tensor_map_[t->name()] = t;
}

namespace session_factory {

typedef std::unordered_map<string, 
                           SessionRegister::Factory> SessionRegistry;
static SessionRegistry* GlobalSessionRegistry() {
  static SessionRegistry* global_session_registry = new SessionRegistry();
  return global_session_registry;
}
void SessionRegister::InitInternal(const string& name, Factory factory) {
  GlobalSessionRegistry()->insert(std::make_pair(name, factory));
}

} //namespace session_factory

SessionBase* GetSession(const string& name) {
  if (session_factory::GlobalSessionRegistry()->count(name) == 0)
    return NULL;
  else
    return session_factory::GlobalSessionRegistry()->at(name)();
}


class SimpleSession : public SessionBase {
 public:
  SimpleSession() {}
  void SetOpChainDef(const OpChainDef& def) override;
  void Run() override;
 private:
  std::vector<std::pair<Op*, OpContext*>> executors_;
};

void SimpleSession::SetOpChainDef(const OpChainDef& def) {
  SessionBase::SetOpChainDef(def);
  for (const OpDef& op_def : op_chain_def_.op()) {
    Op* op = CreateOp(op_def);
    OpContext* context = new OpContext(op_def, this); 
    executors_.push_back(std::make_pair(op, context));
  }
}

void SimpleSession::Run() {
  for (auto& one_pair: executors_) {
    Op* op = one_pair.first;
    OpContext* context = one_pair.second;
    op->Compute(context);
  }
}

REGISTER_SESSION_BUILDER("SimpleSession", SimpleSession);

} //namespace cavs
