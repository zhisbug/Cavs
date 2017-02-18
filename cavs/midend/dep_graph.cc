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

bool DepGraph::TraverseCriticalPath(Scope* loss_scope,
      const Edge* loss, const Edge* curr,
      unordered_map<const Node*, bool>* fwd_path,
      list<const Node*>* newly_traversed) {
  CHECK(curr->srcs_size() == 1) << curr->srcs_size();
  CHECK(curr->dsts_size() <= 1) << curr->dsts_size();
  if (curr == loss ||
      (fwd_path->find(curr->dst(0)) != fwd_path->end() &&
        (*fwd_path)[curr->dst(0)])) {
    for (auto* node : *newly_traversed) {
      loss_scope->AddNode(node->op_def());
    }
    newly_traversed->clear();
    return true;
  }
  CHECK(curr->dsts_size() == 1) << curr->dsts_size();
  const Node* node = curr->dst(0);
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
        for (auto& grad : grads) {
          if (std::find(grad.output().begin(), grad.output().end(),
               OpDecl::GetGradientName(curr->name())) == grad.output().end())
            continue;
          Node* grad_node = loss_scope->AddNode(grad);
          vector<TensorShapeDef> inputs;
          grad_node->InputShapes(&inputs);
          const vector<TensorShapeDef>& shapes = 
            ::backend::ShapeInference(grad, inputs);
          grad_node->SetShape(shapes);
          in_path = true;
          (*fwd_path)[node] = true; 
        }
      }
    }
    if (!in_path)
      newly_traversed->pop_back();
  }
  return (*fwd_path)[node];
}

void DepGraph::GroupClosedSet(
    const vector<string>& vars,
    const Edge* loss,
    const string& solver,
    Scope* loss_scope) {
  unordered_map<const Node*, bool> recalculate;
  for (auto& var_name : vars) {
    const Edge* var = loss_scope->FindEdge(var_name);
    list<const Node*> newly_traversed;
    TraverseCriticalPath(loss_scope, loss, var,
        &recalculate, &newly_traversed);
    OpDef update;  
    ::backend::OpDefBuilder(solver)
        .Input(var_name)
        .Input(OpDecl::GetGradientName(var_name))
        .Output(var_name)
        .Shape(var->shape())
        .Device("GPU")
        .Finalize(&update);
    loss_scope->AddNode(update);
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
  for (auto& attr : def.attr()) {
    if (attr.name() == "vars") {
      auto& vars = attr.value().list().s();
      var_names.resize(vars.size());
      std::copy(vars.begin(), vars.end(), var_names.begin());
    }else if (attr.name() == "solver") {
      solver = attr.value().s(); 
    }else if (attr.name() == "projection") {
      proj = attr.value().s(); 
    }else if (attr.name() == "iters") {
      iters = attr.value().i(); 
    }
  }
  CHECK(var_names.size());
  CHECK(solver.length());
  CHECK(proj.length());
  CHECK(iters > 0);
  Scope* loss_scope = new Scope(s_, def.output(0));
  Edge* loss_edge = s_->FindEdge(loss);
  CHECK(loss_edge);
  OpDef const_op;
  BuildConstantOpDef(&const_op, 
      OpDecl::GetGradientName(loss),
      loss_edge->shape(), 1.f);
  loss_scope->AddNode(const_op);
  GroupClosedSet(var_names, loss_edge, solver, loss_scope);
  ScopedNode* sn = new ScopedNode(iters, loss_scope, def);
  //s_->PrintSymbolTable();
}

void DepGraph::Dump() {
  for (auto* node : s_->nodes_)
    LOG(INFO) << node->op_def().DebugString();
}

} //namespace midend
