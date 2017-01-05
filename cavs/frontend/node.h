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
  void SetShape(const ::midend::TensorShapeDef& def); 
  Node* GetGradientNode();

 private:
  ::midend::OpDef op_def_;
  std::vector<const Node*> inputs_;
  std::vector<const Node*> outputs_;
};

} //namespace frontend

#endif
