#ifndef CAVS_MIDEND_SCOPE_H_
#define CAVS_MIDEND_SCOPE_H_

#include "cavs/midend/dep_graph.h"

namespace midend {

class Scope {
 public:
  Scope(Scope* s, std::string n) : father_(s), name_(n) {}
  Node* FindNode(std::string n);
  void AddNestedScope(Scope* scope);
  void AddNode(Node* node);
    
 private:
  std::string name_;
  Scope* father_;
  std::vector<Scope*> children_;
  std::unordered_map<std::string, Edge*> edge_table_;
  std::unordered_map<std::string, Node*> node_table_;
  std::unordered_map<std::string, NodeGroup*> out2ng_;
};

const Scope* GetGlobalScope();

} //namespace midend
#endif
