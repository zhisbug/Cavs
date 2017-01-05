#include "cavs/frontend/node.h"

namespace frontend {

void Node::InputShapes(
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

void Node::SetShape(const ::midend::TensorShapeDef& def) {
  for (int i = 0; i < op_def_.attr_size(); i++) {
    ::midend::OpDef::AttrDef* attr = op_def_.mutable_attr(i);
    if (attr->name() == "shape") {
      *(attr->mutable_value()->mutable_shape()) = def;
      break;
    }
  }
}

Node* Node::GetGradientNode() {
  Node* grad = new Node(this->op_def_);
  grad->op_def_->set_name(this->op_def_.name()+"_grad");
  grad->op_def_->clear_input();
  for (auto& out : this->op_def_.output())
    grad->op_def_->add_input(out);
  grad->op_def_->clear_output();
  for (auto& inp : this->op_def_.input())
    grad->op_def_->add_output(inp);

} //namespace frontend
