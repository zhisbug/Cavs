#ifndef CAVS_MIDEND_SCOPE_H_
#define CAVS_MIDEND_SCOPE_H_

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
  std::unordered_map<std::string, Node*> node_table_;
};

Scope* GetGlobalScope();

} //namespace midend
#endif
