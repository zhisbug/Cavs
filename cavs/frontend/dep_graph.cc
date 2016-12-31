#include "cavs/frontend/dep_graph.h"
#include "cavs/util/logging.h"

#include <string>

using namespace std;

namespace frontend {

Node* DepGraph::AddNode(const ::midend::OpDef& op_def) {
  const string& out = op_def.output(0);
  CHECK(out2node_.find(out) == out2node_.end());
  Node* node = new Node(op_def);
  out2node_[out] = node;
  nodes_.push_back(node);

  for (auto& input : op_def.input()) {
    CHECK(out2node_.find(input) != out2node_.end());
    node->AddInput(out2node_[input]);
    out2node_[input]->AddOutput(node);
  }
  return node;
}

} //namespace frontend
