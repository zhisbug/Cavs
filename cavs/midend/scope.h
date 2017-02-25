#ifndef CAVS_MIDEND_SCOPE_H_
#define CAVS_MIDEND_SCOPE_H_

#include "cavs/midend/dep_graph.h"
#include "cavs/proto/op_def.pb.h"

#include <string>
#include <set>
#include <unordered_map>

namespace midend {

class Edge;
class Node;
class NodeGroup;
class Scope {
 public:
  Scope(const Scope* s, const std::string& n);
  Scope* FindChild(const std::string& n) const;
  Edge* FindEdge(const std::string& n, bool within = false) const;
  Node* AddNode(const OpDef& op_def);
  void AddNode(const Node* node);
  void AddEdge(const Edge* edge);
  void PrintSymbolTable();
  inline const std::string& name() const {
    return name_; 
  }
  std::string DebugInfo();
  friend class DepGraph;
  friend class ScopedNode;
    
 private:
  const std::string name_;
  const Scope* father_;
  std::unordered_map<std::string, Scope*> children_;
  std::unordered_map<std::string, Edge*> edge_table_;
  std::vector<Node*> nodes_;
  std::unordered_map<std::string, Edge*> in_edges_;
  std::unordered_map<std::string, Edge*> out_edges_;
};

Scope* GetGlobalScope();

} //namespace midend
#endif
