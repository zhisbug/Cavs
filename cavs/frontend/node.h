#ifndef CAVS_FRONTEND_NODE_H_
#define CAVS_FRONTEND_NODE_H_

#include "cavs/midend/op_def.pb.h"

#include <string>
#include <vector>

namespace frontend {

class Node {
 public:
  explicit Node(const ::midend::OpDef& op_def) : op_def_(op_def) {}
  Node* input_node(int idx);
  Node* output_node(int idx);
  void AddInput(const Node* n);
  void AddOutput(const Node* n);

 private:
  ::midend::OpDef op_def_;
  std::vector<Node*> inputs_;
  std::vector<Node*> outputs_;
};

} //namespace frontend

#endif
