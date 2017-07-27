#ifndef CAVS_MIDEND_GRAPH_SESSION_H_
#define CAVS_MIDEND_GRAPH_SESSION_H_

#include "cavs/midend/session_base.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/graph_scheduler.h"

namespace midend {

class SessionBase;

class GraphSession : public SessionBase {
 public:
  GraphSession(SessionBase* sb, const std::string& name, int max_graph_node_count)
    : global_sess_(sb), name_(name), MAX_NODE_(max_graph_node_count) {
    CHECK(name_.length());
    gscheduler_ = new GraphScheduler();
    scope_ = main_scope();
    if (sb->session_type() & FUSION)  this->AddType(FUSION);
  }
  const Tensor* GetTensor(const std::string& name, bool recursive = false) const override;
  OpContext* GetContext(const Node* node) override;
  std::string TensorNameInFunctionContext(const Edge* e) const;
  virtual int session_type() const { return GRAPH; }
  GraphScheduler* graph_scheduler() { return gscheduler_; }

 private:
  SessionBase* global_sess_;
  const Scope* scope_;
  GraphScheduler* gscheduler_;
  const int MAX_NODE_;
  std::string name_;
};

GraphSession* GetGraphSession(const std::string& name);
bool InsertGraphSession(const std::string& name, GraphSession* sess);

} //namespace midend

#endif
