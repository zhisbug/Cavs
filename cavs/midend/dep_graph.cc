#include "cavs/midend/dep_graph.h"
#include "cavs/midend/statement_builder.h"
#include "cavs/util/logging.h"
#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"

#include <string>
#include <algorithm>
#include <list>

using namespace std;
using ::backend::OpDecl;
using ::backend::BuildConstantOpDef;

namespace midend {

Node* DepGraph::AddNode(const OpDef& op_def) { 
  return s_->AddNode(op_def); 
}

const Node* DepGraph::FindNode(
    const std::string& name) const {
  const Edge* edge = s_->FindEdge(name);
  if (!edge) return NULL;
  CHECK(edge->isStateful() || edge->srcs_size() == 1)
    << edge->name() << edge->srcs_size();
  return edge->src(0);
}

const Edge* DepGraph::FindEdge(
    const std::string& name) const {
  return s_->FindEdge(name);
}

bool DepGraph::TraverseCriticalPath(Scope* loss_scope,
      const Edge* loss, const Edge* curr,
      unordered_map<const Node*, bool>* fwd_path,
      list<const Node*>* newly_traversed) {
  CHECK(curr->srcs_size() == 1) << curr->DebugInfo();
  LOG_IF(INFO, curr->dsts_size() > 1) << curr->DebugInfo();
  //LOG(INFO) << curr->DebugInfo();
  if (curr == loss) {
    for (auto* node : *newly_traversed) {
      loss_scope->AddNode(node->op_def());
    }
    newly_traversed->clear();
    return true;
  }
  const Node* node = curr->dst(0);
  CHECK(node);
  if (fwd_path->find(node) != fwd_path->end() && (*fwd_path)[node]) {
    DeduceAndApplyOneGradNode(loss_scope, node, curr->name());
    for (auto* node : *newly_traversed) {
      loss_scope->AddNode(node->op_def());
    }
    newly_traversed->clear();
    return true;
  }
  //CHECK(curr->dsts_size() == 1) << curr->dsts_size();
  if (fwd_path->find(node) == fwd_path->end() ||
      !(*fwd_path)[node]) {
    (*fwd_path)[node] = false;
    newly_traversed->push_back(node);
    bool in_path = false;
    for (auto* edge : node->outputs()) {
      if (TraverseCriticalPath(loss_scope, loss, edge,
            fwd_path, newly_traversed)) {
        const vector<OpDef>& grads = 
          ::backend::MakeGradient(node->op_def()); 
        CHECK(grads.size()) << node->op_def().DebugString();
        //for (auto& grad : grads) {
          //if (std::find(grad.output().begin(), grad.output().end(),
               //GetGradientName(curr->name())) == grad.output().end()) {
            //continue;
          //}
          //Node* grad_node = loss_scope->AddNode(grad);
          //vector<TensorShapeDef> inputs;
          ////LOG(INFO) << grad.DebugString();
          //grad_node->InputShapes(&inputs);
          //const vector<TensorShapeDef>& shapes = 
            //::backend::ShapeInference(grad, inputs);
          //grad_node->SetShape(shapes);
        //}
        DeduceAndApplyOneGradNode(loss_scope, node, curr->name());
        in_path = true;
        (*fwd_path)[node] = true; 
      }
    }
    if (!in_path) {
      CHECK(newly_traversed->size() > 0);
      newly_traversed->pop_back();
    }
  }
  return (*fwd_path)[node];
}

void DepGraph::DeduceAndApplyOneGradNode(
    Scope* s,
    const Node* node,
    const string& edge) {
  const vector<OpDef>& grads = 
    ::backend::MakeGradient(node->op_def()); 
  CHECK(grads.size()) << node->op_def().DebugString();
  bool exist = false;
  for (auto& grad : grads) {
    if (std::find(grad.output().begin(), grad.output().end(),
         GetGradientName(edge)) == grad.output().end()) {
      continue;
    }
    CHECK(!exist) << "Two grad possible?"
                  << node->op_def().DebugString()
                  << grad.DebugString();
    Node* grad_node = s->AddNode(grad);
    if (grad_node) {
      vector<TensorShapeDef> inputs;
      grad_node->InputShapes(&inputs);
      const vector<TensorShapeDef>& shapes = 
        ::backend::ShapeInference(grad, inputs);
      grad_node->SetShape(shapes);
    }
    exist = true;
  }
  CHECK(exist) << "No gradient wrt the edge name";
  return;
}

void DepGraph::GroupClosedSet(
    const vector<string>& vars,
    const Edge* loss,
    const string& solver,
    const float lr,
    const string& proj,
    Scope* loss_scope) {
  unordered_map<const Node*, bool> recalculate;
  for (auto& var_name : vars) {
    LOG(INFO) << var_name;
    const Edge* var = loss_scope->FindEdge(var_name);
    list<const Node*> newly_traversed;
    TraverseCriticalPath(loss_scope, loss, var,
        &recalculate, &newly_traversed);
    OpDef update;  
    ::backend::OpDefBuilder(solver)
        .Input(var_name)
        .Input(GetGradientName(var_name))
        .Output(var_name)
        .Shape(var->shape())
        .AttrSingle<float>("learning_rate", lr)
        .Device("GPU")
        .Finalize(&update);
    loss_scope->AddNode(update);
    if (proj.length() > 0) {
      //LOG(FATAL) << proj;
      OpDef projection;  
      ::backend::OpDefBuilder(proj)
          .Input(var_name)
          .Output(var_name)
          .Shape(var->shape())
          .Device("GPU")
          .Finalize(&projection);
      loss_scope->AddNode(projection);
    }
  }
}

void DepGraph::GroupAllVariables(vector<string>* vars) {
  for (Node* n : s_->nodes_) {
    if (static_cast<SingleNode*>(n)->IsVariableOp()) {
      CHECK(n->outputs_size() == 1) << n->outputs_size();
      vars->push_back(n->output(0)->name());
    }
  }
}

void DepGraph::OptimizeWithLoss(
    const OpDef& def) {
  CHECK(def.input_size() == 1);
  const string& loss = def.input(0);
  vector<string> var_names(0);
  int iters = 0;
  string proj;
  string solver;
  float lr;
  for (auto& attr : def.attr()) {
    if (attr.name() == "Vars") {
      auto& vars = attr.value().list().s();
      var_names.resize(vars.size());
      std::copy(vars.begin(), vars.end(), var_names.begin());
    }else if (attr.name() == "Solver") {
      solver = attr.value().s(); 
    }else if (attr.name() == "Projection") {
      proj = attr.value().s(); 
    }else if (attr.name() == "Iters") {
      iters = attr.value().i(); 
    }else if (attr.name() == "learning_rate") {
      lr = attr.value().f(); 
    }
  }
  CHECK(var_names.size());
  CHECK(solver.length());
  //LOG(INFO) << "Projection Method: " << proj;
  CHECK(iters > 0);
  Scope* loss_scope = new Scope(s_, def.output(0));
  Edge* loss_edge = s_->FindEdge(loss);
  CHECK(loss_edge);
  OpDef const_op;
  BuildConstantOpDef(&const_op, 
      GetGradientName(loss),
      loss_edge->shape(), 1.f);
  loss_scope->AddNode(const_op);
  GroupClosedSet(var_names, loss_edge, solver, lr, proj, loss_scope);
  ScopedNode* sn = new ScopedNode(iters, loss_scope, def);
  //s_->PrintSymbolTable();
}

string DepGraph::DebugInfo() {
  return s_->DebugInfo();
}

} //namespace midend
