#ifndef CAVS_FRONTEND_NODE_H_
#define CAVS_FRONTEND_NODE_H_

#include "cavs/proto/op_def.pb.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>

namespace frontend {

class Node {
 public:
  explicit Node(const OpDef& op_def) : op_def_(op_def) {}
  inline void AddInput(const Node* n) {
    inputs_.push_back(n);
  }
  inline void AddOutput(const Node* n) {
    outputs_.push_back(n);
  }
  void InputShapes(std::vector<const TensorShapeDef*>* inputs);
  inline const OpDef& op_def() const {
    return op_def_; 
  }
  void SetShape(const TensorShapeDef& def); 
  Node* GetGradientNode();

 private:
  OpDef op_def_;
  std::vector<const Node*> inputs_;
  std::vector<const Node*> outputs_;
};

} //namespace frontend

#endif
