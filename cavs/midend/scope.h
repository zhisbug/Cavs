#ifndef CAVS_MIDEND_SCOPE_H_
#define CAVS_MIDEND_SCOPE_H_

#include "cavs/midend/dep_graph.h"
#include "cavs/proto/op_def.pb.h"

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
  Edge* FindEdge(const std::string& n, bool within = false) const;
  Node* AddNode(const OpDef& op_def);
  void AddGradNode(const OpDef& op_def);
  friend class DepGraph;
  //NodeGroup* FindNodeGroup(const std::string& n);
  //void AddNodeGroup(const Edge* edge);
    
 private:
  const std::string name_;
  const Scope* father_;
  std::vector<Scope*> children_;
  std::unordered_map<std::string, Edge*> edge_table_;
  std::vector<Node*> nodes_;
  std::unordered_map<std::string, Edge*> in_edges_;
  std::unordered_map<std::string, Edge*> out_edges_;
  void AddEdge(Edge* edge);
  //std::unordered_map<std::string, NodeGroup*> nodegroup_table_;
  //Node* FindNode(const std::string& n, bool within = false) const;
};

const Scope* GetGlobalScope();

} //namespace midend
#endif
