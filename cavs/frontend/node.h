#ifndef CAVS_FRONTEND_NODE_H_
#define CAVS_FRONTEND_NODE_H_

#include "cavs/midend/op_def.pb.h"
#include "cavs/midend/tensor_shape.pb.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>

namespace frontend {

class Node {
 public:
  explicit Node(const ::midend::OpDef& op_def) : op_def_(op_def) {}
  void AddInput(const Node* n) { inputs_.push_back(n); }
  void AddOutput(const Node* n) { outputs_.push_back(n); }
  void InputShapes(std::vector<const ::midend::TensorShapeDef*>* inputs);
  //Node* input_node(int idx);
  //Node* output_node(int idx);

 private:
  ::midend::OpDef op_def_;
  std::vector<const Node*> inputs_;
  std::vector<const Node*> outputs_;
};

__inline__ void Node::InputShapes(
    std::vector<const ::midend::TensorShapeDef*>* inputs ) {
  for (auto* node : inputs_) {
    for (auto& attr : node->op_def_.attr()) {
      if (attr.name() == "shape") {
        CHECK(attr.value().has_shape());
        inputs->push_back(&(attr.value().shape()));
        break;
      }
    }
  }
}

} //namespace frontend

#endif
