#ifndef CAVS_MIDEND_GRAPH_SESSION_H_
#define CAVS_MIDEND_GRAPH_SESSION_H_

#include "cavs/midend/session_base.h"
#include "cavs/midend/tensor.h"

namespace midend {

class GraphSession : public SessionBase {
 public:
  GraphSession(SessionBase* s) : global_sess_(s) {}
  const Tensor* GetTensor(const std::string& name, bool recursive = false) const override;
  OpContext* GetContext(const Node* node) override;
  virtual int session_type() const { return GRAPH; }

 private:
  SessionBase* global_sess_;
};

} //namespace midend

#endif
