#ifndef CAVS_FRONTEND_DEP_GRAPH_H_
#define CAVS_FRONTEND_DEP_GRAPH_H_

#include "cavs/frontend/node.h"
#include "cavs/proto/op_def.pb.h"

#include <vector>
#include <unordered_map>

namespace frontend {

class DepGraph {
 public:
  Node* AddNode(const OpDef& op_def);
  inline int num_nodes() const {
    return nodes_.size();
  }
  inline const Node* operator[](int node_id) const {
    return nodes_[node_id];
  }
  void GradientPass();

 private:
  std::vector<Node*> nodes_;
  std::unordered_map<std::string, Node*> out2node_;
};

} //namespace frontend

#endif
