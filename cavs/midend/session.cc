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

SimpleSession::SimpleSession(const OpChainDef& def) 
    : SessionBase(def) {
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

} //namespace cavs
