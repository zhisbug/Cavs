#include "cavs/midend/session.h"
#include "cavs/midend/allocator.h"
#include "cavs/midend/op.h"
#include "cavs/util/logging.h"

namespace cavs {

const Tensor* SessionBase::GetTensor(const string& name) const {
  if (tensor_map_.count(name) == 0)
    return NULL;
  else
    return &(tensor_map_.at(name));
}

void SessionBase::InsertTensor(const Tensor& t){
  CHECK(tensor_map_.count(t.name()) == 0);
  tensor_map_[t.name()] = t;
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

SessionBase* GetSession(const string& name, const OpChainDef& def) {
  if (session_factory::GlobalSessionRegistry()->count(name) == 0)
    return NULL;
  else
    return session_factory::GlobalSessionRegistry()->at(name)(def);
}


class SimpleSession : public SessionBase {
 public:
  //SimpleSession() {}
  SimpleSession(const OpChainDef& def) override;
  void Run(const vector<string>& output_names, 
           vector<const Tensor*>* output_tensors,
           const vector<string>& input_names,
           const vector<const Tensor*>& input_tensors) override;
 private:
  void FeedInput(const vector<string>& input_names,
                 const vector<const Tensor*>& input_tensors) override;
  std::vector<std::pair<Op*, OpContext*>> executors_;
};

SimpleSession::SimpleSession(const OpChainDef& def) {
  for (const OpDef& op_def : op_chain_def_.op()) {
    Op* op = CreateOp(op_def);
    OpContext* context = new OpContext(op_def, this); 
    executors_.push_back(std::make_pair(op, context));
  }
}

void SimpleSession::Run(const vector<string>& output_names,
    vector<const Tensor*>* output_tensors,
    const vector<string>& input_names,
    const vector<const Tensor*>& input_tensors) {
  FeedInput(input_names, input_tensors);
  for (auto& one_pair : executors_) {
    Op* op = one_pair.first;
    OpContext* context = one_pair.second;
    op->Compute(context);
  }

  for (int i = 0; i < output_names.size(); i++) {
    (*output_tensors)[i] = GetTensor(output_names[i]);
    CHECK_NOTNULL((*output_tensors)[i]);
  }
}

void SimpleSession::FeedInput(const vector<string>& input_names,
    const vector<const Tensor*>& input_tensors) {
  CHECK(input_names.size() == input_tensors.size());
  for (int i = 0; i < input_names.size(); i++) {
    tensor_map[input_names[i]] = *input_tensors[i];
  }
}

REGISTER_SESSION_BUILDER("SimpleSession", SimpleSession);

} //namespace cavs
