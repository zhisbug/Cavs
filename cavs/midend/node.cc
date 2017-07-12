#include "cavs/midend/node.h"
#include "cavs/midend/graph_session.h"

using std::string;
using std::vector;

namespace midend {

Node::Node(Scope* located) : 
  located_(located), inputs_(0), outputs_(0), stmt_(NULL) {
  located->AddNode(this);
}

string Node::scoped_name() const {
  CHECK(scope());
  return scope()->scoped_name() + ":" + name();
}

void Node::AddInput(const Edge* e) {
  //CHECK(e->scope() == scope());
  inputs_.push_back(const_cast<Edge*>(e));
}

void Node::AddOutput(const Edge* e) {
  CHECK(e->scope() == scope() || e->isVariable());
  outputs_.push_back(const_cast<Edge*>(e));
}

void Node::AddControlDependency(const Edge* e) {
  CHECK(e->scope() == scope());
  control_dependency_.push_back(const_cast<Edge*>(e));
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

string SingleNode::debug_info() const {
  return "\nname:\t" + name() +
         "(scope: " + located_->name() + ")\n" +
         op_def_.DebugString();
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
  if (!stmt_) {
    OpImpl* op = CreateOp(op_def());
    OpContext* ctxt = sess->GetContext(this);

    VLOG(V_DEBUG) << "Compiling GraphNode:\t" << op_def().name();
    CHECK(!gsess_);
    int max_graph_node_count = GetSingleArg<int>(op_def_, "MaxGraphNodeCount");
    CHECK(max_graph_node_count > 0);
    GraphScheduler* gs = new GraphScheduler();
    gsess_ = new GraphSession(sess, located_, gs, max_graph_node_count);
    //gsess_->SetOutputTensor(ctxt->Output(0));

    //Scope* leaf = main_scope()->FindChildScope("Leaf");
    //CHECK_NOTNULL(leaf);
    ////we should add the generated scopednode into main_scope()
    ////because only main_scope may not be wrapped up into a ScopedNode and executed.
    //ScopedNode* lsn = new ScopedNode(main_scope(), "Leaf", 1);
    //lsn->SetContainedScope(leaf);
    //Statement* lstmt = lsn->Compile(gsess_);

    //Scope* inode = main_scope()->FindChildScope("Inode");
    //CHECK_NOTNULL(inode);
    //ScopedNode* isn = new ScopedNode(main_scope(), "Inode", 1);
    //isn->SetContainedScope(inode);
    //Statement* istmt = isn->Compile(gsess_);
    
    Scope* node_func = main_scope()->FindChildScope("Node");
    CHECK_NOTNULL(node_func);
    ScopedNode* sn = new ScopedNode(main_scope(), "Node", 1);
    sn->SetContainedScope(node_func);
    Statement* node_func_stmt = sn->Compile(gsess_);

    ctxt->SetGraphScheduler(gs);
    stmt_ = new GraphStatement(node_func_stmt, gs);
    dynamic_cast<GraphStatement*>(stmt_)->SetOp(op);
    dynamic_cast<GraphStatement*>(stmt_)->SetContext(ctxt);
  }
  return stmt_;
}

Statement* GraphGradNode::Compile(
    SessionBase* sess) {
  if (!stmt_) {
    OpImpl* op = CreateOp(op_def());
    OpContext* ctxt = sess->GetContext(this);

    VLOG(V_DEBUG) << "Compiling GraphGradNode:\t" << op_def().name();
    CHECK_NOTNULL(forward_node_);
    //when the graphgrad node is compiled,
    //the graph node must have been compile already
    //that means its graph session has been set
    gsess_ = forward_node_->gsess_;
    CHECK_NOTNULL(gsess_);
    //gsess_->SetOutputGradTensor(ctxt->inputs_(0));

    //Scope* leaf = main_scope()->FindChildScope("Leaf")->FindChildScope(GetGradientName("Leaf"));
    //CHECK_NOTNULL(leaf);
    ////we should add the generated scopednode into main_scope()
    ////because only main_scope may not be wrapped up into a ScopedNode and executed.
    //ScopedNode* lsn = new ScopedNode(main_scope(), GetGradientName("Leaf"), 1);
    //lsn->SetContainedScope(leaf);
    //Statement* lstmt = lsn->Compile(gsess_);

    //Scope* inode = main_scope()->FindChildScope("Inode")->FindChildScope(GetGradientName("Inode"));
    //CHECK_NOTNULL(inode);
    //ScopedNode* isn = new ScopedNode(main_scope(), GetGradientName("Inode"), 1);
    //isn->SetContainedScope(inode);
    //Statement* istmt = isn->Compile(gsess_);
    
    Scope* node_grad_func = main_scope()->FindChildScope("Node")->FindChildScope(GetGradientName("Node"));
    CHECK_NOTNULL(node_grad_func);
    ScopedNode* sn = new ScopedNode(main_scope(), GetGradientName("Node"), 1);
    sn->SetContainedScope(node_grad_func);
    Statement* node_grad_stmt = sn->Compile(gsess_);

    ctxt->SetGraphScheduler(gsess_->graph_scheduler());
    stmt_ = new GraphGradStatement(node_grad_stmt, gsess_->graph_scheduler());
    dynamic_cast<GraphStatement*>(stmt_)->SetOp(op);
    dynamic_cast<GraphStatement*>(stmt_)->SetContext(ctxt);
  }
  return stmt_;
}

ScopedNode::ScopedNode(Scope* located,
      const string& name, int iter)
    : Node(located), name_(name), iter_(iter), contained_(NULL) {}

void ScopedNode::SetContainedScope(const Scope* contained) {
  CHECK_NOTNULL(contained);
  CHECK(!contained_);
  contained_ = contained;
  for (auto& edge: contained->in_edges_) {
    inputs_.push_back(edge.second);
  }

  Edge* output = new Edge(name_, located_);
  output->AddSource(this);
  nodes_.assign(contained_->typological_sorted_nodes_.begin(),
                contained_->typological_sorted_nodes_.end());
}

Statement* ScopedNode::Compile(
    SessionBase* sess) {
  CHECK_NOTNULL(contained_);
  if (!stmt_) {
    VLOG(V_DEBUG) << "Compiling ScopeNode:\t"  << scoped_name();
    VLOG(V_DEBUG) << "It is located in scope " << scope()->scoped_name();
    VLOG(V_DEBUG) << "It contains a scope "    << contained_->scoped_name();
    BasicBlock* bb = new BasicBlock(iter_);
    for (auto* node : nodes_) {
      VLOG(V_DEBUG) << "\tCompiling\t" << node->name()
                    << "\t in Scope: " << contained_->scoped_name();
      Statement* stmt = node->Compile(sess);
      CHECK(stmt) << node->debug_info();
      bb->AppendStmt(stmt);
    }
    stmt_ = bb;
  }
  return stmt_;
}

string ScopedNode::debug_info() const {
  return "\nname:\t" + name() +
         "(scope: " + located_->name() + ")\n" +
         contained_->debug_info();
}

} //namespace midend
