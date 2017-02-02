#include "cavs/midend/dep_graph.h"
#include "cavs/midend/statement_builder.h"
#include "cavs/util/logging.h"
#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"

#include <string>
#include <algorithm>

using namespace std;
using ::backend::OpDecl;

namespace midend {

//DepGraph::DepGraph() {
  ////OpDef def;
  ////def.set_name("__internal__sink");
  ////sink_ = AddNode(def, GetGlobalScope());
//}

//void DepGraph::AddGradNode(const OpDef& op_def,
    //const Scope* s) {
//}

Node* DepGraph::AddNode(const OpDef& op_def) { 
  return s_->AddNode(op_def); 
}

bool DepGraph::SearchCriticalPath(Scope*s,
      const Edge* var, const Edge* curr,
      const unordered_map<Node*, bool>& recalculate,
      const unordered_map<Node*, bool>& accessed) {
  if (curr == var) return true;
  if (curr->isStateful()) return false;
  CHECK(curr->srcs_size() == 1);
  const Node* node = curr->src[0];
  if (recalculate.find(node) == recalculate.end() &&
      accessed.find(node) == accessed.end()) {
    accessed[node] = true; 
    for (auto* edge : node->inputs()) {
      if (SearchCriticalPath(s, var, edge, recalculate)) {
        s->AddNode(child->op_def(), true);
        recalculate[node] = true; 
      }
    }
  }
  return recalculate.find(node) != recalculate.end();
}

void DepGraph::SearchClosedSet(
    const vector<string>& vars,
    const vector<string>& grad_vars,
    Scope* s) {
  CHECK(vars.size() == grad_vars.size());
  unordered_map<Node*, bool> recalculate;
  for (int i = 0; i < vars.size(); i++) {
    const Edge* var = s->FindEdge(vars[i]);
    const Edge* grad_var = s->FindEdge(grad_vars[i]);
    unordered_map<Node*, bool> accessed;
    SearchCriticalPath(s, var, grad_var,
        &recalculate, &accessed);
  }
  //const vector<Edge*>& input_edges = father->inputs_;
  //if (input_edges.size() == 0) {
    //*contained = false;
    //return;
  //}else {
    //bool contain_child = false;
    //for (auto* edge : input_edges) {
      //CHECK(edge->src_.size() == 1);
      //SearchClosedSet(edge->src_[0], ng, vars, &contain_child);
      //if (std::find(vars.begin(), vars.end(), edge->name())
          //!= vars.end()) {
        //contain_child = true;
      //}
    //}
    //if (contain_child) {
      //*contained = true;
      //for (auto* edge : input_edges) {
        //CHECK(edge->src_.size() == 1);
        //ng->AddNode(edge->src_[0]);
      //}
    //}
  //}
}

void DepGraph::GroupAllVariables(vector<string>* vars) {
  for (Node* n : nodes_) {
    if (n->IsVariableOp())
      vars->push_back(n->name());
  }
}

void DepGraph::OptimizeWithLoss(
    const string& loss, 
    const string& solver, 
    const vector<string>& var_names) {
  CHECK(var_names.size() > 0);
  Scope* s_loss = new Scope(GetGlobalScope(), loss);
  BackPropagate();
  vector<string> grad_var_names;
  for (auto& var : var_names)
    grad_var_names.push_back(OpDecl::GetGradientName(var));
  Edge* loss_edge = GetGlobalScope()->FindEdge(loss);
  CHECK(loss_node);
  OpDef const_op;
  BuildConstantOpDef(&const_op, 
      OpDecl::GetGradientName(loss),
      loss_edge->shape(),
      1.f);
  AddNode(const_op, s_loss);
  SearchClosedSet(var_names, grad_var_names, s_loss);
  
  //unordered_map<string, bool> calculated_edges;
  //vector<Statement*> ordered_statements;
  for (auto& var : var_names) {
    const string& var_grad = 
      OpDecl::GetGradientName(var);
    //const Edge* root = out2edge_[var_grad];
    //RecursiveSearchInputNode(root, &ordered_statements);
  }
  //AddGradNode(op_def);
  //return BuildBasicBlock(ordered_statements);
}

//currently, we assume all ops defined by developpers
//are in the scope of global
void DepGraph::BackPropagate() {
  for (int i = grad_nodes_.size();
        i < nodes_.size(); i++) {
    AddGradNode(nodes_[i]->op_def(), GetGlobalScope());  
  }
  ////gen_grads->clear();
  for (int i = num_nodes()-1; i >= 0; i--) {
    const vector<OpDef>& grads = 
      ::backend::MakeGradient(nodes_[i]->op_def()); 
    for (auto& grad : grads) {
      ////for (auto& grad_out_str : grad.output())
        ////gen_grads->push_back(grad_out_str);
      //for (auto& grad_input : grad.input()) {
        ////if the grad_input does not exist, 
        ////it must be the loss node,
        ////and it should be set to one-value matrix
        //if (out2edge_.find(grad_input) == out2edge_.end()) {
          //const string& ori_input = 
            //::backend::OpDecl::GetOriginName(grad_input);
          //CHECK(out2edge_.find(ori_input) != out2edge_.end());
          //OpDef const_op;
          //::backend::BuildConstantOpDef(&const_op, 
              //grad_input,
              //out2edge_[ori_input]->tensor_shape_,
              //1.f);
          //AddNode(const_op);
        //}
      //}
      Node* node = AddNode(grad);
      vector<TensorShapeDef> inputs;
      node->InputShapes(&inputs);
      const vector<TensorShapeDef>& out_shapes = 
        ::backend::ShapeInference(grad, inputs);
      node->SetShape(out_shapes);
    }
  }
}

void DepGraph::AddSolver(
    const string& solver,
    const vector<string>& var_names,
    vector<Statement*>* stmts) {
  //for (auto& var : var_names) {
    //CHECK(out2edge_.find(var) != out2edge_.end());
    //const string& var_grad = 
      //::backend::OpDecl::GetGradientName(var);
    //CHECK(out2edge_.find(var_grad) != out2edge_.end());
    //OpDef update;  
    //::backend::OpDefBuilder(solver)
        //.Input(var_grad)
        //.Output(var)
        //.Shape(out2edge_.at(var)->tensor_shape_)
        //.Finalize(&update);
    ////Node* node = AddNode(update);
    //stmts->emplace_back(BuildExprStatement(update));
  //}
}

void NodeGroup::AddNode(const Node* n) {
  for (auto* node : nodes_)
    CHECK(node != n);
  nodes_.push_back(n);
  for (auto* edge : n->inputs()) {
    auto it = std::find(outputs_.begin(), outputs_.end(), edge);
    if (it == outputs_.end()) {
      if (std::find(inputs_.begin(), inputs_.end(), edge) 
          == inputs_.end()) {
        inputs_.push_back(edge);
      }
    }else {
      outputs_.erase(it);
    }
  }
  for (auto* edge : n->outputs()) {
    auto it = std::find(inputs_.begin(), inputs_.end(), edge); 
    if (it == inputs_.end()) {
      if (std::find(outputs_.begin(), outputs_.end(), edge) 
          == outputs_.end()) {
        outputs_.push_back(edge); 
      }
    }else {
      inputs_.erase(it);  
    }
  }
}


void DepGraph::SetLossNodeGroup(const string& loss,
    const vector<string>& vars,
    const Scope* s) {
  //CHECK(s.Fine(loss) == out2ng_.end());
  //NodeGroup* ng = new NodeGroup(loss); 
}

void DepGraph::Dump() {
  for (auto* node : nodes_)
    LOG(INFO) << node->op_def().DebugString();
}

} //namespace midend
