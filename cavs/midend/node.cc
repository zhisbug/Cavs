#include "cavs/midend/node.h"
#include "cavs/midend/graph_session.h"
#include "cavs/midend/runtime_compiler/code_generator.h"
#include "cavs/midend/stream_scheduler.h"
#include "cavs/midend/batch_weight_updater.h"
#include "cavs/util/op_def_builder.h"

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
  : Node(s), op_def_(op_def), isDynamicEnabled_(false), sess_debug_(NULL) {}

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

void SingleNode::SetDynamicEnabled() {
  //currently, only single-output node can be batched
  CHECK(output_size() == 1);
  isDynamicEnabled_ = true;
  output(0)->SetDynamicEnabled();
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
    if ((sess->session_type() & SessionBase::MPI) &&
        (op_def().name() == "Variable" ||
         op_def().name() == "DDV" ||
         op_def().name() == "Data")) {
      OpDef mpi_def = op_def();
      mpi_def.set_name(op_def().name()+"MPI");
      LOG(INFO) << "Compiling SingleNode:\t" << mpi_def.name();
      VLOG(V_DEBUG) << mpi_def.DebugString();
      op = CreateOp(mpi_def);
    }else {
      LOG(INFO) << "Compiling SingleNode:\t" << op_def().DebugString();
      VLOG(V_DEBUG) << op_def().DebugString();
      op = CreateOp(op_def());
    }
    OpContext* ctxt = sess->GetContext(this);
    CHECK(ctxt) << op_def().DebugString();
    CHECK(op) << op_def().DebugString();
    CHECK(ctxt) << op_def().DebugString();
    ExprStatement* expr_stmt = new ExprStatement(op, ctxt);
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
    OpContext* ctxt = sess->GetContext(this);
    ExprStatement* push_arg_stmt = NULL;
    ExprStatement* pop_ret_stmt = NULL;
    OpDef push_arg_def;
    OpDefBuilder("FunctionPushArg")
      .Input(this->input(1)->name())
      .Device("GPU")
      .Finalize(&push_arg_def);
    OpImpl *push_arg_op = CreateOp(push_arg_def);
    OpDef pop_ret_def;
    OpDefBuilder("FunctionPopRet")
      .Output(this->output(0)->name())
      .Device("GPU")
      .Finalize(&pop_ret_def);
    OpImpl *pop_ret_op = CreateOp(pop_ret_def);
    OpContext* push_ctxt = ctxt->ExtractContext({1}, {});
    OpContext* pop_ctxt  = ctxt->ExtractContext({}, {0});
    

    VLOG(V_DEBUG) << "Compiling GraphNode:\t" << op_def().name();
    if (!(gsess_ = GetGraphSession(op_def_.output(0)))) {
      int max_graph_node_count = GetSingleArg<int>(op_def_, "MaxGraphNodeCount");
      CHECK(max_graph_node_count > 0);
      //GraphScheduler* gs = new GraphScheduler();
      gsess_ = new GraphSession(sess, op_def_.output(0), max_graph_node_count);
      InsertGraphSession(op_def_.output(0), gsess_);
    }

    ScopedNode* sn = dynamic_cast<ScopedNode*>(main_scope()->FindNode("Node"));
    bool pop_exist = false;
    if (!sn) {
      Scope* node_func = main_scope()->FindChildScope("Node");
      CHECK_NOTNULL(node_func);
      sn = new ScopedNode(main_scope(), "Node", 1);
      sn->SetContainedScope(node_func);
      for (Node* n : sn->nodes_) {
        if (n->name() == "Push") {
          pop_exist = true; 
          break;
        }
      }
    }
    CHECK_NOTNULL(gsess_);
    Statement* node_func_stmt = sn->Compile(gsess_);

    push_ctxt->SetGraphScheduler(gsess_->graph_scheduler());
    push_arg_stmt = new ExprStatement(push_arg_op, push_ctxt);
    stmt_ = new GraphStatement(node_func_stmt, gsess_->graph_scheduler());
    dynamic_cast<GraphStatement*>(stmt_)->SetGlobalContext(ctxt);
    dynamic_cast<GraphStatement*>(stmt_)->SetPushArgStatement(push_arg_stmt);
    if (pop_exist) {
      pop_ctxt->SetGraphScheduler(gsess_->graph_scheduler());
      pop_ret_stmt = new ExprStatement(pop_ret_op, pop_ctxt);
      dynamic_cast<GraphStatement*>(stmt_)->SetPopRetStatement(pop_ret_stmt);
    }
  }
  return stmt_;
}

GraphGradNode::GraphGradNode(const OpDef& op_def, Scope* s)
  : SingleNode(op_def, s), gsess_(NULL) {}

Statement* GraphGradNode::Compile(
    SessionBase* sess) {
  if (!stmt_) {
    //OpImpl* op = CreateOp(op_def());
    //OpContext* ctxt = sess->GetContext(this);
    OpContext* ctxt = sess->GetContext(this);
    ExprStatement* push_arg_stmt = NULL;
    ExprStatement* pop_ret_stmt = NULL;
    OpDef push_arg_def;
    OpDefBuilder("FunctionPushArg")
      .Input(this->input(0)->name())
      .Device("GPU")
      .Finalize(&push_arg_def);
    OpImpl *push_arg_op = CreateOp(push_arg_def);
    OpDef pop_ret_def;
    OpDefBuilder("FunctionPopRet")
      .Output(this->output(0)->name())
      .Device("GPU")
      .Finalize(&pop_ret_def);
    OpImpl* pop_ret_op   = CreateOp(pop_ret_def);
    OpContext* push_ctxt = ctxt->ExtractContext({0}, {});
    OpContext* pop_ctxt  = ctxt->ExtractContext({}, {0});

    VLOG(V_DEBUG) << "Compiling GraphGradNode:\t" << op_def().name();
    //when the graphgrad node is compiled,
    //the graph node must have been compile already
    //that means its graph session has been set
    CHECK_NOTNULL(gsess_ = GetGraphSession(GetOriginName(op_def_.input(0))));
    
    CHECK(main_scope()->FindChildScope("Node"));
    bool pop_exist = false;
    ScopedNode* sn = dynamic_cast<ScopedNode*>(main_scope()->FindChildScope("Node")->FindNode(GetGradientName("Node")));
    if (!sn){
      Scope* node_grad_func = main_scope()->FindChildScope("Node")->FindChildScope(GetGradientName("Node"));
      CHECK_NOTNULL(node_grad_func);
      sn = new ScopedNode(main_scope(), GetGradientName("Node"), 1);
      sn->SetContainedScope(node_grad_func);
      for (Node* n : sn->nodes_) {
        if (n->name() == "Push") {
          pop_exist = true; 
          break;
        }
      }
    }

    vector<Statement*> batch_weight_update;
    std::list<Node*> finalize_node;
    if ((sess->opt_type() & OPT_BATCHING)) {
      VLOG(V_DEBUG) << "Begin modifing the critical path for Batching in ScopedNode";
      BatchingWeightUpdater updater(&(sn->nodes_), &finalize_node);
      VLOG(V_DEBUG) << "Modifing the critical path done for Batching in ScopedNode";
    }

    Statement* node_grad_stmt = sn->Compile(gsess_);

    if (sess->opt_type() & OPT_BATCHING) {
      for (Node* fn : finalize_node) {
        Statement* stmt = fn->Compile(gsess_);
        CHECK(stmt) << fn->debug_info();
        batch_weight_update.push_back(stmt);
      }
    }

    push_ctxt->SetGraphScheduler(gsess_->graph_scheduler());
    push_arg_stmt = new ExprStatement(push_arg_op, push_ctxt);
    stmt_ = new GraphGradStatement(node_grad_stmt, gsess_->graph_scheduler());
    dynamic_cast<GraphGradStatement*>(stmt_)->SetGlobalContext(ctxt);
    dynamic_cast<GraphGradStatement*>(stmt_)->SetPushArgStatement(push_arg_stmt);
    if (pop_exist) {
      pop_ctxt->SetGraphScheduler(gsess_->graph_scheduler());
      pop_ret_stmt = new ExprStatement(pop_ret_op, pop_ctxt);
      dynamic_cast<GraphGradStatement*>(stmt_)->SetPopRetStatement(pop_ret_stmt);
    }
    if (!batch_weight_update.empty())
      dynamic_cast<GraphGradStatement*>(stmt_)->SetBatchWeightUpdate(std::move(batch_weight_update));
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

    if ((sess->opt_type() & OPT_FUSION) && sess->session_type() == SessionBase::GRAPH) {
      VLOG(V_DEBUG) << "Begin modifing the critical path for fusion in ScopedNode";
      RTC::CodeGenerator generator(&nodes_);
      VLOG(V_DEBUG) << "Modifing the critical path done for fusion in ScopedNode";
    }

    for (auto* node : nodes_) {
      VLOG(V_DEBUG) << "\tCompiling\t" << node->name()
                    << "\t in Scope: " << contained_->scoped_name();
      Statement* stmt = node->Compile(sess);
      CHECK(stmt) << node->debug_info();
      bb->AppendStmt(stmt);
    }

    if ((sess->opt_type() & OPT_STREAMMING) && sess->session_type() == SessionBase::GRAPH) {
      VLOG(V_DEBUG) << "Begin modifing the critical path for streamming in ScopedNode";
      //if (dependency.empty()) {
        //StreamScheduler::DependencyExtractor(&dependency, nodes_);
      //}
      //CHECK(dependency.size() == nodes_.size());
      StreamScheduler scheduler(&(bb->stmts_), nodes_);
      VLOG(V_DEBUG) << "Modifing the critical path done for streamming in ScopedNode";
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
