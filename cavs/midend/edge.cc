#include "cavs/midend/edge.h"

using std::string;

namespace midend {

Edge::Edge(const string& name, bool stateful, Scope* s)
  : stateful_(stateful), s_(s),
    name_(name),
    scoped_name_(s->name() + ":" + name) {
  s_->AddEdge(this);
}

void Edge::AddDst(Node* node) {
  dsts_.push_back(node); 
}

void Edge::AddSource(Node* node) {
  CHECK(stateful_ || srcs_.empty())
    << node->DebugInfo()
    << DebugInfo();
  srcs_.push_back(node); 
}

string Edge::DebugInfo() const {
  return "\nname:\t" + name() + 
         "\nshape:\t" + shape().DebugString() + 
         "scope:\t" + s_->name() +
         "\tsrcs_size:\t" + std::to_string(srcs_size()) + 
         "\ndsts_size:\t" + std::to_string(dsts_size());
}

} //namespace midend
