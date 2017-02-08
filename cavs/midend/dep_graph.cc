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
  return const_cast<Scope*>(s_)->AddNode(op_def); 
}

int DepGraph::num_nodes() const {
  return s_->nodes_.size();
}

const Node* DepGraph::operator[](int node_id) const {
  return s_->nodes_[node_id];
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
          Node* grad_node = const_cast<Scope*>(loss_scope)->AddNode(grad);
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
        .Finalize(&update);
    const_cast<Scope*>(loss_scope)->AddNode(update);
  }
}

void DepGraph::GroupAllVariables(vector<string>* vars) {
  for (Node* n : s_->nodes_) {
    if (n->IsVariableOp()) {
      CHECK(n->outputs_size() == 1);
      vars->push_back(n->output(0)->name());
    }
  }
}

void DepGraph::OptimizeWithLoss(
    const string& loss, 
    const string& solver, 
    const vector<string>& var_names) {
  CHECK(var_names.size() > 0);
  Scope* loss_scope = new Scope(GetGlobalScope(), loss);
  Edge* loss_edge = s_->FindEdge(loss);
  CHECK(loss_edge);
  OpDef const_op;
  BuildConstantOpDef(&const_op, 
      OpDecl::GetGradientName(loss),
      loss_edge->shape(),
      1.f);
  loss_scope->AddNode(const_op);
  GroupClosedSet(var_names, loss_edge, solver, loss_scope);
}

//currently, we assume all ops defined by developpers
//are in the scope of global
void DepGraph::BackPropagate() {
}

//void DepGraph::SetLossNodeGroup(const string& loss,
    //const vector<string>& vars,
    //const Scope* s) {
  ////CHECK(s.Fine(loss) == out2ng_.end());
  ////NodeGroup* ng = new NodeGroup(loss); 
//}

void DepGraph::Dump() {
  for (auto* node : s_->nodes_)
    LOG(INFO) << node->op_def().DebugString();
}

} //namespace midend
