#include "cavs/frontend/dep_graph.h"
#include "cavs/util/logging.h"

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

void Node::InputShapes(
    vector<TensorShapeDef>* inputs) {
  for (auto* edge: inputs_) {
    inputs->push_back(edge->shape());
  }
}

//void Node::SetShape(const vector<TensorShapeDef>& def) {
  //for (auto& shape_def : def)
    //*(op_def_.add_shape()) = shape_def;
//}

} //namespace frontend
