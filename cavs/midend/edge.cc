#include "cavs/midend/edge.h"

using std::string;

namespace midend {

Edge::Edge(const string& name, bool stateful, Scope* s)
  : tensor_name_(name), stateful_(stateful), s_(s) {
  s_->AddEdge(this);
}

void Edge::AddDst(Node* node) {
  dsts_.push_back(node); 
}

string Edge::DebugInfo() const {
  return "\nname:\t" + name() + 
         "\nshape:\t" + shape().DebugString() + 
         "\nscope:\t" + s_->name() +
         "srcs_size:\t" + std::to_string(srcs_size()) + 
         "\ndsts_size:\t" + std::to_string(dsts_size());
}

} //namespace midend
