#include "scope.h"

namespace midend {

Node* Scope::FindNode(std::string n) {
  Scope* s = this;
  while (s && s->node_table_.find(n) == s->node_table_.end()) {
    s = s->father_; 
  }
  if (s)
    return s->node_table_.at(n);
  else
    return NULL;
}

void AddNestedScope(Scope* scope) {
  children_.push_back(scope);
}

void AddNode(Node* node) {
  const string& name = node->op_def()->name();
  CHECK(node_table_.find(name) ==
        node_table_.end());
  node_table[name] = node;
}

const Scope* Scope::GetGlobalScope() {
  static Scope* s = new Scope(NULL, "global");
  return s;
}

} //namespace midend
