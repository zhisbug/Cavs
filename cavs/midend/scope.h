#ifndef CAVS_MIDEND_SCOPE_H_
#define CAVS_MIDEND_SCOPE_H_

#include "cavs/midend/dep_graph.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace midend {

class Edge;
class Node;
class NodeGroup;
class Scope {
 public:
  Scope(const Scope* s, const std::string& n)
    : father_(s), name_(n) {}
  Node* FindNode(const std::string& n) const;
  Edge* FindEdge(const std::string& n) const;
  void AddNode(Node* node);
  void AddEdge(Edge* edge);
  //NodeGroup* FindNodeGroup(const std::string& n);
  //void AddNodeGroup(const Edge* edge);
    
 private:
  const std::string name_;
  const Scope* father_;
  std::vector<Scope*> children_;
  std::unordered_map<std::string, Edge*> edge_table_;
  std::unordered_map<std::string, Node*> node_table_;
  std::unordered_map<std::string, NodeGroup*> nodegroup_table_;
};

const Scope* GetGlobalScope();

} //namespace midend
#endif
