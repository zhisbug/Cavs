#include "cavs/frontend/dep_graph.h"
#include "cavs/util/logging.h"
#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_def_builder.h"

#include <string>

using namespace std;

namespace frontend {

Node* DepGraph::AddNode(const OpDef& op_def) {
  Node* node = new Node(op_def);
  nodes_.push_back(node);
  for (auto& out : op_def.output()) {
    CHECK(out2edge_.find(out) == out2edge_.end());
    Edge* out_edge = new Edge(out);
    out2edge_[out] = out_edge;
    out_edge->SetSource(node);
    node->AddOutput(out_edge);
  }

  for (auto& input : op_def.input()) {
    CHECK(out2edge_.find(input) != out2edge_.end());
    node->AddInput(out2edge_[input]);
    out2edge_[input]->AddDst(node);
  }
  return node;
}

void DepGraph::BackPropagate(
    vector<string>* gen_grads, const string& loss) {
  gen_grads->clear();
  for (int i = num_nodes()-1; i >= 0; i--) {
    const vector<OpDef>& grads = 
      ::backend::MakeGradient(nodes_[i]->op_def_); 
    for (auto& grad : grads) {
      //it is only a temporary version
      for (auto& grad_out_str : grad.output())
        gen_grads->push_back(grad_out_str);
      for (auto& grad_input : grad.input()) {
        //if the grad_input does not exist, 
        //it must be the loss node,
        //and it should be set to one-value matrix
        if (out2edge_.find(grad_input) == out2edge_.end()) {
          const string& ori_input = 
            ::backend::OpDecl::GetOriginName(grad_input);
          CHECK(out2edge_.find(ori_input) != out2edge_.end());
          OpDef const_op;
          ::backend::BuildConstantOpDef(&const_op, 
              grad_input,
              out2edge_[ori_input]->tensor_shape_,
              1.f);
          AddNode(const_op);
        }
      }
      Node* node = AddNode(grad);
      vector<TensorShapeDef> inputs;
      node->InputShapes(&inputs);
      const vector<TensorShapeDef>& out_shapes = 
        ::backend::ShapeInference(grad, inputs);
      node->SetShape(out_shapes);
    }
  }
}

void DepGraph::AddSolver(const string& solver) {
  for (int i = 0; i < num_nodes(); i++) {
    if (nodes_[i]->IsVariableOp()) {
      CHECK(nodes_[i]->outputs_.size() == 1);   
      const string& var_name = 
        nodes_[i]->outputs_[0]->tensor_name_;
      CHECK(out2edge_.find(var_name) != out2edge_.end());
      const string& var_grad_name = 
        ::backend::OpDecl::GetGradientName(var_name);
      if (out2edge_.find(var_grad_name) != out2edge_.end()) {
        OpDef update;  
        ::backend::OpDefBuilder(solver)
            .Input(var_grad_name)
            .Output(var_name)
            .Shape(out2edge_.at(var_name)->tensor_shape_)
            .Finalize(&update);
        Node* node = AddNode(update);
      }
    }
  }
}

void DepGraph::DebugString() {
  for (auto* node : nodes_)
    LOG(INFO) << node->op_def_.DebugString();
}

void Node::InputShapes(
    vector<TensorShapeDef>* inputs) {
  for (auto* edge: inputs_) {
    inputs->push_back(edge->shape());
  }
}

} //namespace frontend
