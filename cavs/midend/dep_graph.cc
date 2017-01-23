#include "cavs/midend/dep_graph.h"
#include "cavs/midend/statement_builder.h"
#include "cavs/util/logging.h"
#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"

#include <string>
#include <algorithm>

using namespace std;

namespace midend {

Node* DepGraph::AddNode(const OpDef& op_def) {
  Node* node = new Node(op_def);
  nodes_.push_back(node);
  for (auto& out : op_def.output()) {
    if (out2edge_.find(out) == out2edge_.end()) {
      Edge* new_edge = new Edge(out);
      out2edge_[out] = new_edge;
    }
    Edge* out_edge = out2edge_[out];
    out_edge->AddSource(node);
    node->AddOutput(out_edge);
  }

  for (auto& input : op_def.input()) {
    CHECK(out2edge_.find(input) != out2edge_.end());
    node->AddInput(out2edge_[input]);
    out2edge_[input]->AddDst(node);
  }
  LOG(INFO) << "here";

  return node;
}

void DepGraph::AddGradNode(const OpDef& op_def) {
  const vector<OpDef>& grads = 
    ::backend::MakeGradient(op_def); 
  vector<Node*> grad_vec;
  for (auto& grad : grads) {
    Node* node = new Node(grad);
    grad_vec.push_back(node);
    for (auto& dx : op_def.output()) {
      const string& x = 
        ::backend::OpDecl::GetOriginName(dx);
      CHECK(out2edge_.find(x) != out2edge_.end());
      if (out2edge_.find(dx) == out2edge_.end()) {
        Edge* dx_edge = new Edge(dx);
        out2edge_[dx] = dx_edge;
      }
      Edge* dx_edge = out2edge_[dx];
      dx_edge->AddSource(node);
      node->AddOutput(dx_edge);
    }

    for (auto& dy: op_def.input()) {
      const string& y = 
        ::backend::OpDecl::GetOriginName(dy);
      CHECK(out2edge_.find(y) != out2edge_.end());
      CHECK(out2edge_.find(dy) == out2edge_.end());
      Edge* dy_edge = new Edge(dy);
      out2edge_[dy] = dy_edge;
      node->AddInput(dy_edge);
      dy_edge->AddDst(node);
    }
  }
  grad_nodes_.push_back(std::move(grad_vec));
}

void DepGraph::SearchClosedSet(
    const Node* father,
    NodeGroup* ng,
    const vector<string>& vars,
    bool* contained) {
  const vector<Edge*>& input_edges = father->inputs_;
  if (input_edges.size() == 0) {
    *contained = false;
    return;
  }else {
    bool contain_child = false;
    for (auto* edge : input_edges) {
      CHECK(edge->src_.size() == 1);
      if (std::find(vars.begin(), vars.end(), edge->name())
          != vars.end()) {
        contain_child = true;
        continue;
      }
      SearchClosedSet(edge->src_[0], ng, vars, &contain_child);
    }
    if (contain_child) {
      *contained = true;
      for (auto* edge : input_edges) {
        CHECK(edge->src_.size() == 1);
        ng->AddNode(edge->src_[0]);
      }
    }
  }
}

BasicBlock* DepGraph::OptimizeWithLoss(
    const string& loss, 
    const string& solver, 
    const vector<string>& var_names) {
  unordered_map<string, bool> calculated_edges;
  vector<Statement*> ordered_statements;
  for (auto& var : var_names) {
    const string& var_grad = 
      backend::OpDecl::GetGradientName(var);
    const Edge* root = out2edge_[var_grad];
    //RecursiveSearchInputNode(root, &ordered_statements);
  }
  //AddGradNode(op_def);
  return BuildBasicBlock(ordered_statements);
}

void DepGraph::BackPropagate() {
  for (int i = grad_nodes_.size();
        i < nodes_.size(); i++) {
    AddGradNode(nodes_[i]->op_def());  
  }
  //gen_grads->clear();
  //for (int i = num_nodes()-1; i >= 0; i--) {
    //const vector<OpDef>& grads = 
      //::backend::MakeGradient(nodes_[i]->op_def_); 
    //for (auto& grad : grads) {
      ////it is only a temporary version
      //for (auto& grad_out_str : grad.output())
        //gen_grads->push_back(grad_out_str);
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
      //Node* node = AddNode(grad);
      //vector<TensorShapeDef> inputs;
      //node->InputShapes(&inputs);
      //const vector<TensorShapeDef>& out_shapes = 
        //::backend::ShapeInference(grad, inputs);
      //node->SetShape(out_shapes);
    //}
  //}
}

void DepGraph::AddSolver(
    const string& solver,
    const vector<string>& var_names,
    vector<Statement*>* stmts) {
  for (auto& var : var_names) {
    CHECK(out2edge_.find(var) != out2edge_.end());
    const string& var_grad = 
      ::backend::OpDecl::GetGradientName(var);
    CHECK(out2edge_.find(var_grad) != out2edge_.end());
    OpDef update;  
    ::backend::OpDefBuilder(solver)
        .Input(var_grad)
        .Output(var)
        .Shape(out2edge_.at(var)->tensor_shape_)
        .Finalize(&update);
    //Node* node = AddNode(update);
    stmts->emplace_back(BuildExprStatement(update));
  }
}

void NodeGroup::AddNode(const Node* n) {
  for (auto* node : nodes_)
    CHECK(node != n);
  nodes_.push_back(n);
  for (auto* edge : n->inputs_) {
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
  for (auto* edge : n->outputs_) {
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
    const vector<string>& vars) {
  CHECK(out2ng_.find(loss) == out2ng_.end());
  NodeGroup* ng = new NodeGroup(loss); 
}

void DepGraph::Dump() {
  for (auto* node : nodes_)
    LOG(INFO) << node->op_def_.DebugString();
}

void Node::InputShapes(
    vector<TensorShapeDef>* inputs) {
  for (auto* edge: inputs_) {
    inputs->push_back(edge->shape());
  }
}

} //namespace midend
