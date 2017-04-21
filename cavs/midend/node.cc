#include "cavs/midend/node.h"

using std::string;

namespace midend {

Node::Node(const OpDef& op_def, const Scope* s)
  : op_def_(op_def), located_(const_cast<Scope*>(s)) {
  located_->AddNode(this);
  node_name_ = s->name() + ":" + op_def_.name();
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
  LOG(INFO) << "Compiling SingleNode:\t" << op_def().name();
  //LOG(INFO) << DebugInfo();
  OpImpl* op = NULL;
  if (sess->SessionType() == SessionBase::MPI &&
      op_def().name() == "Variable") {
    OpDef mpi_def = op_def();
    mpi_def.set_name("VariableMPI");
    op = CreateOp(mpi_def);
    LOG(INFO) << mpi_def.DebugString();
  }else {
    op = CreateOp(op_def());
  }
  OpContext* ctxt = sess->GetContext(this);
  CHECK(op) << op_def().DebugString();
  CHECK(ctxt) << op_def().DebugString();
  ExprStatement* expr_stmt =  new ExprStatement(op, ctxt);
  CHECK(expr_stmt);
  return expr_stmt;
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
  for (auto* node_ptr : contained_->nodes_)
    nodes_.push_back(node_ptr);
}

Statement* ScopedNode::Compile(
    SessionBase* sess) const {
  LOG(INFO) << "Compiling ScopeNode:\t" << op_def().output(0);
  LOG(INFO) << "It is located in scope " << scope()->name();
  LOG(INFO) << "It contains a scope " << contained_->name();
  BasicBlock* bb = new BasicBlock(iter_);
  //for (auto* node : contained_->nodes_) {
    //LOG(INFO) << node->op_def().DebugString();
  //}
  for (auto* node : nodes_) {
    //LOG(INFO) << "\tCompiling\t" << node->op_def().name()
              //<< "\t in Scope: " << contained_->name();
    Statement* stmt = node->Compile(sess);
    CHECK(stmt) << node->DebugInfo();
    bb->AppendStmt(stmt);
  }
  return bb;
}

} //namespace midend
