#include "cavs/midend/node.h"

namespace midend {

void Node::SetShape(
    const std::vector<TensorShapeDef>& def) {
  op_def_.clear_shape();
  for (int i = 0; i < outputs_.size(); i++) {
    outputs_[i]->SetShape(def[i]);
    *(op_def_.add_shape()) = def[i];
  }
}

void Node::InputShapes(
    std::vector<TensorShapeDef>* inputs) {
  for (auto* edge: inputs_) {
    inputs->push_back(edge->shape());
  }
}

} //namespace midend
