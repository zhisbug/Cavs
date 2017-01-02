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
  inline void AddInput(const Node* n) {
    inputs_.push_back(n);
  }
  inline void AddOutput(const Node* n) {
    outputs_.push_back(n);
  }
  void InputShapes(std::vector<const ::midend::TensorShapeDef*>* inputs);
  inline const ::midend::OpDef& op_def() const {
    return op_def_; 
  }
  inline void SetShape(const ::midend::TensorShapeDef& def) {
    for (int i = 0; i < op_def_.attr_size(); i++) {
      ::midend::OpDef::AttrDef* attr = op_def_.mutable_attr(i);
      if (attr->name() == "shape") {
        *(attr->mutable_value()->mutable_shape()) = def;
        break;
      }
    }
  }

 private:
  ::midend::OpDef op_def_;
  std::vector<const Node*> inputs_;
  std::vector<const Node*> outputs_;
};

inline void Node::InputShapes(
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
