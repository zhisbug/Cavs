#include "cavs/midend/edge.h"

using std::string;

namespace midend {

Edge::Edge(const string& name, Scope* s)
  : name_(name), located_(s), 
    scoped_name_(s->name() + ":" + name) {
  located_->AddEdge(this);
}

void Edge::AddDst(Node* node) {
  dsts_.push_back(node); 
}

void Edge::AddSource(Node* node) {
  CHECK(isStateful() || srcs_.empty())
    << node->DebugInfo()
    << DebugInfo();
  srcs_.push_back(node); 
}

string Edge::DebugInfo() const {
  return "\nname:\t" + scoped_name() + 
         "\nshape:\t" + shape().DebugString() + 
         "scope:\t" + located_->name() +
         "\tsrcs_size:\t" + std::to_string(srcs_size()) + 
         "\ndsts_size:\t" + std::to_string(dsts_size());
}

} //namespace midend
