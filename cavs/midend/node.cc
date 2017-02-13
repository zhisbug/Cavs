#include "cavs/midend/node.h"

namespace midend {

void SingleNode::SetShape(
    const std::vector<TensorShapeDef>& def) {
  op_def_.clear_shape();
  for (int i = 0; i < outputs_.size(); i++) {
    outputs_[i]->SetShape(def[i]);
    *(op_def_.add_shape()) = def[i];
  }
}

void SingleNode::InputShapes(
    std::vector<TensorShapeDef>* inputs) {
  for (auto* edge: inputs_) {
    inputs->push_back(edge->shape());
  }
}

Statement* SingleNode::Compile(SessionBase* sess) override {
  OpImpl* op = CreateOp(node->op_def());
  OpContext* context = sess->GetContext(node->op_def());
  return new ExprStatement(op, context);
}

void ScopedNode::ScopedNode(int iter, const OpDef& op_def,
    const Scope* s = GetGlobalScope())
    : Node(op_def, s), iter_(iter) {
  for (auto& edge: in_edges_) {
    inputs_.push_back(edge.second);
  }
}

Statement* ScopedNode::Compile(SessionBase* sess) override {
  BasicBlock* bb = new BasicBlock(iter_);
  for (auto* node : s_->nodes_) {
    OpImpl* op = CreateOp(node->op_def());     
    OpContext* ctxt = sess->GetContext(node->op_def());
    bb->AppendStmt(new ExprStatement(op_context));
  }
  return bb;
}

} //namespace midend
