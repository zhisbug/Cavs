#include "scope.h"

using std::string;

namespace midend {

Node* Scope::FindNode(const string& n) const {
  const Scope* s = this;
  while (s && s->node_table_.find(n) == s->node_table_.end()) {
    s = s->father_; 
  }
  if (s)
    return s->node_table_.at(n);
  else
    return NULL;
}

Edge* Scope::FindEdge(const string& n) const {
  const Scope* s = this;
  while (s && s->edge_table_.find(n) == s->edge_table_.end()) {
    s = s->father_;
  }
  if (s)
    return s->edge_table_.at(n);
  else
    return NULL;
}

//void AddNestedScope(Scope* scope) {
  //children_.push_back(scope);
//}

void Scope::AddNode(Node* node) {
  const string& name = node->name();
  CHECK(node_table_.find(name) ==
        node_table_.end());
  node_table_[name] = node;
}

void Scope::AddEdge(Edge* edge) {
  const string& name = edge->name();
  CHECK(edge_table_.find(name) ==
        edge_table_.end());
  edge_table_[name] = edge;
}

const Scope* GetGlobalScope() {
  static Scope* s = new Scope(NULL, "global");
  return s;
}

} //namespace midend
