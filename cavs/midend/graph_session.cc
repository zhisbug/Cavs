#include "cavs/midend/graph_session.h"

using std::string;

namespace midend {

const Tensor* GraphSession::GetTensor(const string& name, bool recursive) const {
  const Tensor* t;
  if (t = SessionBase::GetTensor(name, recursive))
    return t;
  else if (t = global_sess_->GetTensor(name, recursive))
    return t;
  else 
    return NULL;
}

OpContext* GraphSession::GetContext(const Node* node) {
  //currently, we use the base class method
  return SessionBase::GetContext(node);
}

} //namespace midend
