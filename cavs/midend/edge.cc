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

} //namespace midend
