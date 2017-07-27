#include "cavs/midend/simple_session.h"
#include "cavs/midend/statement.h"
#include "cavs/midend/runtime_compiler/code_generator.h"

using std::string;
using std::vector;
using std::list;
using std::set;

namespace midend {

class FusionSession: public SimpleSession {
 public:
  int session_type() const override { return FUSION; }
 private:
  void Compile(const vector<string>& output_names) override;
};

void FusionSession::Compile(
    const vector<string>& output_names) {
  list<Node*> critical_path;
  set<Node*> include;
  for (auto& output : output_names) {
    Node* node = const_cast<Node*>(s_->FindNode(output));
    CHECK(node);
    DepthSearch(node, &critical_path, &include);
  }

  VLOG(V_DEBUG) << "Begin modifing the critical path";
  RTC::CodeGenerator generator(&critical_path);
  VLOG(V_DEBUG) << "Modifing the critical path done";

  CHECK(executors_.find(HashString(output_names)) == executors_.end());
  vector<Statement*>* executor = &executors_[HashString(output_names)];
  for (auto* node : critical_path) {
    LOG(INFO) << node->debug_info();
    LOG(INFO) << node->scope()->name();
    LOG(INFO) << "compiling\t" << node->name();
    Statement* stmt = node->Compile(this);
    CHECK(stmt);
    executor->push_back(stmt);
  }
  return;
}

REGISTER_SESSION_BUILDER("FusionSession", FusionSession);

} //namespace midend


