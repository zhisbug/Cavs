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

Statement* SingleNode::Compile(
    SessionBase* sess) const {
  OpImpl* op = CreateOp(op_def());
  OpContext* ctxt = sess->GetContext(op_def());
  return new ExprStatement(op, ctxt);
}

ScopedNode::ScopedNode(int iter,
      const OpDef& op_def,
      const Scope* s)
    : Node(op_def, s), iter_(iter) {
  for (auto& edge: s->in_edges_)
    inputs_.push_back(edge.second);
}

Statement* ScopedNode::Compile(
    SessionBase* sess) const {
  BasicBlock* bb = new BasicBlock(iter_);
  for (auto* node : s_->nodes_) {
    OpImpl* op = CreateOp(node->op_def());     
    OpContext* ctxt = sess->GetContext(node->op_def());
    bb->AppendStmt(new ExprStatement(op, ctxt));
  }
  return bb;
}

} //namespace midend
