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
  : Node(s), op_def_(op_def), sess_debug_(NULL) {}

void SingleNode::SetShape(
    const vector<TensorShapeDef>& def) {
  CHECK(def.size() == outputs_.size())
      << "in shapes:\t" << def.size()
      << "\tneeded shapes:\t" << outputs_.size()
      << "\nnode info: " << op_def_.DebugString();
  op_def_.clear_shape();
  for (int i = 0; i < outputs_.size(); i++) {
    outputs_[i]->SetShape(def[i]);
    *(op_def_.add_shape()) = def[i];
  }
}

Statement* SingleNode::Compile(
    SessionBase* sess) {
  if (!stmt_) {
    {
      CHECK(sess && (!sess_debug_ || sess_debug_ == sess)) 
          << "currently, we only support one node is compiled by one session";
      sess_debug_ = sess;
    }
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
    CHECK(ctxt) << op_def().DebugString();
    CHECK(op) << op_def().DebugString();
    CHECK(ctxt) << op_def().DebugString();
    ExprStatement* expr_stmt =  new ExprStatement(op, ctxt);
    CHECK(expr_stmt);
    stmt_ = expr_stmt;
  }
  return stmt_;
}

GraphNode::GraphNode(const OpDef& op_def, Scope* s)
  : SingleNode(op_def, s), gsess_(NULL) {}

Statement* GraphNode::Compile(
    SessionBase* sess) {
  if (!stmt_) {
    OpImpl* op = CreateOp(op_def());
    OpContext* ctxt = sess->GetContext(this);

    VLOG(V_DEBUG) << "Compiling GraphNode:\t" << op_def().name();
    //CHECK(!gsess_);
    //int max_graph_node_count = GetSingleArg<int>(op_def_, "MaxGraphNodeCount");
    //CHECK(max_graph_node_count > 0);
    //GraphScheduler* gs = new GraphScheduler();
    //gsess_ = new GraphSession(sess, located_, gs, max_graph_node_count);
    ////gsess_->SetOutputTensor(ctxt->Output(0));
    if (!(gsess_ = GetGraphSession(op_def_.output(0)))) {
      int max_graph_node_count = GetSingleArg<int>(op_def_, "MaxGraphNodeCount");
      CHECK(max_graph_node_count > 0);
      //GraphScheduler* gs = new GraphScheduler();
      gsess_ = new GraphSession(sess, op_def_.output(0), max_graph_node_count);
      InsertGraphSession(op_def_.output(0), gsess_);
    }

    ScopedNode* sn = dynamic_cast<ScopedNode*>(main_scope()->FindNode("Node"));
    if (!sn) {
      Scope* node_func = main_scope()->FindChildScope("Node");
      CHECK_NOTNULL(node_func);
      sn = new ScopedNode(main_scope(), "Node", 1);
      sn->SetContainedScope(node_func);
    }
    CHECK_NOTNULL(gsess_);
    Statement* node_func_stmt = sn->Compile(gsess_);

    ctxt->SetGraphScheduler(gsess_->graph_scheduler());
    stmt_ = new GraphStatement(node_func_stmt, gsess_->graph_scheduler());
    dynamic_cast<GraphStatement*>(stmt_)->SetOp(op);
    dynamic_cast<GraphStatement*>(stmt_)->SetContext(ctxt);
  }
  return stmt_;
}

GraphGradNode::GraphGradNode(const OpDef& op_def, Scope* s)
  : SingleNode(op_def, s), gsess_(NULL) {}

Statement* GraphGradNode::Compile(
    SessionBase* sess) {
  if (!stmt_) {
    OpImpl* op = CreateOp(op_def());
    OpContext* ctxt = sess->GetContext(this);

    VLOG(V_DEBUG) << "Compiling GraphGradNode:\t" << op_def().name();
    //CHECK_NOTNULL(forward_node_);
    //when the graphgrad node is compiled,
    //the graph node must have been compile already
    //that means its graph session has been set
    //gsess_ = forward_node_->gsess_;
    //CHECK_NOTNULL(gsess_);
    //gsess_->SetOutputGradTensor(ctxt->inputs_(0));

    VLOG(V_DEBUG) << "here";
    CHECK_NOTNULL(gsess_ = GetGraphSession(GetOriginName(op_def_.input(0))));
    
    CHECK(main_scope()->FindChildScope("Node"));
    //CHECK(!main_scope()->FindChildScope("Node")->FindNode(GetGradientName("Node"))->IsSingleNode());
    ScopedNode* sn = dynamic_cast<ScopedNode*>(main_scope()->FindChildScope("Node")->FindNode(GetGradientName("Node")));
    VLOG(V_DEBUG) << "here";
    if (!sn){
      Scope* node_grad_func = main_scope()->FindChildScope("Node")->FindChildScope(GetGradientName("Node"));
      CHECK_NOTNULL(node_grad_func);
      sn = new ScopedNode(main_scope(), GetGradientName("Node"), 1);
      sn->SetContainedScope(node_grad_func);
    }
    VLOG(V_DEBUG) << "here";
    Statement* node_grad_stmt = sn->Compile(gsess_);

    ctxt->SetGraphScheduler(gsess_->graph_scheduler());
    stmt_ = new GraphGradStatement(node_grad_stmt, gsess_->graph_scheduler());
    VLOG(V_DEBUG) << "here";
    dynamic_cast<GraphStatement*>(stmt_)->SetOp(op);
    dynamic_cast<GraphStatement*>(stmt_)->SetContext(ctxt);
    VLOG(V_DEBUG) << "here";
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
