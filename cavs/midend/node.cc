#include "cavs/midend/node.h"

using std::string;
using std::vector;

namespace midend {

//Node::Node(const OpDef& op_def, Scope* s)
  //: op_def_(op_def), located_(s), stmt_(NULL){
  //located_->AddNode(this);
//}

Node::Node(Scope* located) : 
  located_(located), inputs_(0), outputs_(0), stmt_(NULL) {
  located->AddNode(this);
}

string Node::scoped_name() const {
  CHECK(scope());
  return scope()->name() + ":" + name();
}

void Node::AddInput(const Edge* e) {
  //CHECK(e->scope() == scope());
  inputs_.push_back(const_cast<Edge*>(e));
}

void Node::AddOutput(const Edge* e) {
  CHECK(e->scope() == scope() || e->isVariable());
  outputs_.push_back(const_cast<Edge*>(e));
}

vector<TensorShapeDef> Node::input_shapes() const {
  vector<TensorShapeDef> ret;
  for (auto* edge: inputs_) {
    CHECK(edge->shape().dim_size() > 0);
    ret.push_back(edge->shape());
  }
  return ret;
}

string Node::debug_info() const {
  return "\nname:\t" + name() +
         "(scope: " + located_->name() + ")";
}

SingleNode::SingleNode(const OpDef& op_def, Scope* s)
  : Node(s), op_def_(op_def) {
}

void SingleNode::SetShape(
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

Statement* SingleNode::Compile(
    SessionBase* sess) {
  if (!stmt_) {
    //LOG(INFO) << DebugInfo();
    OpImpl* op = NULL;
    if (sess->session_type() == SessionBase::MPI &&
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

Statement* GraphNode::Compile(
    SessionBase* sess) {
  //if (!stmt_) {
    ////Scope*
    //OpImpl* op = NULL;
    //VLOG(V_DEBUG) << "Compiling GraphNode:\t" << op_def().name();
    //op = CreateOp(op_def());
    //OpContext* ctxt = sess->GetContext(this);
    //CHECK(op) << op_def().DebugString();
    //CHECK(ctxt) << op_def().DebugString();
    //GraphStatement* graph_stmt =  new GraphStatement(op, ctxt);
    //CHECK(expr_stmt);
    //stmt_ = expr_stmt;
  //}
  //return stmt_;
}

ScopedNode::ScopedNode(Scope* located,
      const Scope* contained, const string& name, int iter)
    : Node(located), name_(name), iter_(iter), contained_(contained) {
  for (auto& edge: contained->in_edges_) {
    inputs_.push_back(edge.second);
  }
  //CHECK(op_def.output_size() == 1);
  //Edge* output = new Edge(op_def.output(0), located_);
  Edge* output = new Edge(name, located_);
  output->AddSource(this);
  nodes_.assign(contained_->typological_sorted_nodes_.begin(),
                contained_->typological_sorted_nodes_.end());
}

Statement* ScopedNode::Compile(
    SessionBase* sess) {
  if (!stmt_) {
    VLOG(V_DEBUG) << "Compiling ScopeNode:\t"  << name_;
    VLOG(V_DEBUG) << "It is located in scope " << scope()->name();
    VLOG(V_DEBUG) << "It contains a scope "    << contained_->name();
    BasicBlock* bb = new BasicBlock(iter_);
    for (auto* node : nodes_) {
      VLOG(V_DEBUG) << "\tCompiling\t" << name()
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
