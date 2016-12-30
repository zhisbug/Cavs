#ifndef CAVS_FRONTEND_DEP_GRAPH_H_
#define CAVS_FRONTEND_DEP_GRAPH_H_

#include "cavs/midend/op_def.pb.h"
#include "cavs/frontend/node.h"

#include <vector>
#include <unordered_map>

namespace frontend {

class DepGraph {
 public:
  void AddNode(const Node* node) {}

 private:
  std::vector<Node*> nodes_;
  std::unordered_map<std::string, Node*> out2node_;
};

} //namespace frontend

#endif
