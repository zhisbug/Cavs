#include "cavs/midend/node.h"

using std::string;

namespace midend {

Node::Node(const OpDef& op_def, Scope* s)
  : op_def_(op_def), located_(s) {
    located_->AddNode(this);
}

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
    CHECK(edge->shape().dim_size() > 0);
    inputs->push_back(edge->shape());
  }
}

string Node::DebugInfo() const {
  return "\nname:\t" + op_def_.name() +
         "(scope: " + located_->name() + ")" +
         "\noutput[0]:\t" + op_def_.output(0);
}

Statement* SingleNode::Compile(
    SessionBase* sess) const {
  OpImpl* op = CreateOp(op_def());
  OpContext* ctxt = sess->GetContext(op_def());
  CHECK(op) << op_def().DebugString();
  CHECK(ctxt) << op_def().DebugString();
  return new ExprStatement(op, ctxt);
}

ScopedNode::ScopedNode(int iter,
      Scope* contained,
      const OpDef& op_def,
      Scope* located)
    : iter_(iter), contained_(contained),
      Node(op_def, located) {
  for (auto& edge: contained->in_edges_) {
    inputs_.push_back(edge.second);
  }
  CHECK(op_def.output_size() == 1);
  Edge* output =
    new Edge(op_def.output(0), false, located_);
  output->AddSource(this);
}

Statement* ScopedNode::Compile(
    SessionBase* sess) const {
  BasicBlock* bb = new BasicBlock(iter_);
  //for (auto* node : contained_->nodes_) {
    //LOG(INFO) << node->op_def().DebugString();
  //}
  for (auto* node : contained_->nodes_) {
    OpImpl* op = CreateOp(node->op_def());     
    OpContext* ctxt = sess->GetContext(node->op_def());
    CHECK(op) << node->op_def().DebugString();
    CHECK(ctxt) << node->op_def().DebugString();
    bb->AppendStmt(new ExprStatement(op, ctxt));
  }
  return bb;
}

} //namespace midend
