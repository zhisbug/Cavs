#include "cavs/frontend/dep_graph.h"
#include "cavs/util/logging.h"
#include "cavs/backend/op_decl.h"

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

void DepGraph::BackPropagate(const string& loss) {
  for (int i = num_nodes()-1; i >= 0; i--) {
    const vector<OpDef>& grads = 
      ::backend::MakeGradient(nodes_[i]->op_def()); 
    for (auto& grad : grads) {
      Node* node = AddNode(grad);
      vector<TensorShapeDef> inputs;
      node->InputShapes(&inputs);
      const vector<TensorShapeDef>& out_shapes = 
        ::backend::ShapeInference(grad, inputs);
      node->SetShape(out_shapes);
    }
  }
}

void Node::InputShapes(
    vector<TensorShapeDef>* inputs) {
  for (auto* edge: inputs_) {
    inputs->push_back(edge->shape());
  }
}

} //namespace frontend
