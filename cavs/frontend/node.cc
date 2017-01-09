#include "cavs/frontend/node.h"

namespace frontend {

void Node::InputShapes(
    std::vector<const TensorShapeDef*>* inputs ) {
  for (auto* node : inputs_) {
    inputs->push_back(&(node->op_def_.shape()));
  }
}

void Node::SetShape(const TensorShapeDef& def) {
  *(op_def_.mutable_shape()) = def;
}

//Node* Node::GetGradientNode() {
  //Node* grad = new Node(this->op_def_);
  //grad->op_def_.set_name(this->op_def_.name()+"_grad");
  //grad->op_def_.clear_input();
  //for (auto& out : this->op_def_.output())
    //grad->op_def_.add_input(out);
  //grad->op_def_.clear_output();
  //for (auto& inp : this->op_def_.input())
    //grad->op_def_.add_output(inp);
//}

} //namespace frontend
