#include "cavs/midend/node.h"

using std::string;
using std::vector;

namespace midend {

std::string Node::name() const {
  return node_name_;
}

std::string Node::scoped_name() const {
  return located_->name() + ":" + node_name_;
}

Node::Node(const OpDef& op_def, Scope* s)
  : op_def_(op_def), located_(s), stmt_(NULL){
  located_->AddNode(this);
  node_name_ = op_def_.name();
}

void Node::AddInput(const Edge* e) {
  //CHECK(e->scope() == scope());
  inputs_.push_back(const_cast<Edge*>(e));
}

void Node::AddOutput(const Edge* e) {
  CHECK(e->scope() == scope() || e->isStateful());
  outputs_.push_back(const_cast<Edge*>(e));
}


void Node::SetShape(
    const vector<TensorShapeDef>& def) {
  CHECK(def.size() == outputs_.size())
      << "in shapes:\t" << def.size()
      << "\tneeded shapes:\t" << outputs_.size();
  op_def_.clear_shape();
  for (int i = 0; i < outputs_.size(); i++) {
    outputs_[i]->SetShape(def[i]);
    *(op_def_.add_shape()) = def[i];
  }
}

//void Node::InputShapes(
    //std::vector<TensorShapeDef>* inputs) {
vector<TensorShapeDef> Node::input_shapes() {
  vector<TensorShapeDef> ret;
  for (auto* edge: inputs_) {
    CHECK(edge->shape().dim_size() > 0);
    ret.push_back(edge->shape());
  }
  return ret;
}

string Node::debug_info() const {
  return "\nname:\t" + op_def_.name() +
         "(scope: " + located_->name() + ")" +
         "\noutput[0]:\t" + op_def_.output(0);
}

Statement* SingleNode::Compile(
    SessionBase* sess) {
  if (!stmt_) {
    //LOG(INFO) << DebugInfo();
    OpImpl* op = NULL;
    if (sess->SessionType() == SessionBase::MPI &&
        (op_def().name() == "Variable" ||
         op_def().name() == "DDV" ||
         op_def().name() == "Data")) {
      OpDef mpi_def = op_def();
      mpi_def.set_name(op_def().name()+"MPI");
      LOG(INFO) << "Compiling SingleNode:\t" << mpi_def.name();
      VLOG(V_DEBUG) << mpi_def.DebugString();
      op = CreateOp(mpi_def);
    }else {
      LOG(INFO) << "Compiling SingleNode:\t" << op_def().name();
      VLOG(V_DEBUG) << op_def().DebugString();
      op = CreateOp(op_def());
    }
    OpContext* ctxt = sess->GetContext(this);
    CHECK(op) << op_def().DebugString();
    CHECK(ctxt) << op_def().DebugString();
    ExprStatement* expr_stmt =  new ExprStatement(op, ctxt);
    CHECK(expr_stmt);
    stmt_ = expr_stmt;
  }
  return stmt_;
}

ScopedNode::ScopedNode(Scope* located, const Scope* contained,
      const OpDef& op_def, int iter)
    : iter_(iter), contained_(contained), Node(op_def, located) {
  for (auto& edge: contained->in_edges_) {
    inputs_.push_back(edge.second);
  }
  CHECK(op_def.output_size() == 1);
  Edge* output = new Edge(op_def.output(0), located_);
  output->AddSource(this);
  nodes_.assign(contained_->typological_sorted_nodes_.begin(),
                contained_->typological_sorted_nodes_.end());
}

Statement* ScopedNode::Compile(
    SessionBase* sess) {
  if (!stmt_) {
    VLOG(V_DEBUG) << "Compiling ScopeNode:\t"  << op_def().output(0);
    VLOG(V_DEBUG) << "It is located in scope " << scope()->name();
    VLOG(V_DEBUG) << "It contains a scope "    << contained_->name();
    BasicBlock* bb = new BasicBlock(iter_);
    for (auto* node : nodes_) {
      VLOG(V_DEBUG) << "\tCompiling\t" << node->op_def().name()
                    << "\t in Scope: " << contained_->name();
      Statement* stmt = node->Compile(sess);
      CHECK(stmt) << node->debug_info();
      bb->AppendStmt(stmt);
    }
    stmt_ = bb;
  }
  return stmt_;
}

} //namespace midend
