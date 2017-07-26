#include "cavs/midend/runtime_compiler/parser.h"
#include "cavs/util/op_def_builder.h"
#include "cavs/util/logging.h"

#include <algorithm>

using std::list;
using std::vector;
using std::set;
using std::string;
using std::unordered_map;

namespace midend {
namespace RTC {

bool isElementwise(string op) {
  static vector<string> elementwise_ops = {"Add", "Minus", "Mul"};
  return (std::find(elementwise_ops.begin(), elementwise_ops.end(), op)
        != elementwise_ops.end());
}

bool isFusable(Node* node) {
  return node->IsSingleNode() && isElementwise(node->name());
}

Parser::Parser(list<Node*>* n) : nodes_(n) {
  group_.resize(nodes_->size(), 0);
  auto iter = nodes_->begin();
  for (int i = 0; i < nodes_->size(); i++, iter++) {
    group_[i] = i; 
    CHECK(node2idx_.find(*iter) == node2idx_.end());
    node2idx_[*iter] = i;
  }
}

int Parser::FindGroup(int id) const{
  CHECK(id < group_.size());
  int parent_id = group_[id];
  while (parent_id != group_[parent_id])
    parent_id = group_[parent_id];
  return parent_id;
}

int Parser::GenerateGroup() {
  auto iter = nodes_->begin();
  for (int i = 0; i < nodes_->size(); i++, iter++) {
    if (isFusable(*iter)) {
      CHECK((*iter)->output_size() == 1);
      Edge* edge = (*iter)->output(0);
      Node* parent_node = edge->dst(0, true);
      CHECK(node2idx_.find(parent_node) != node2idx_.end());
      if (isFusable(parent_node)) {
        group_[i] = FindGroup(node2idx_.at(parent_node));
      }
      for (int j = 1; j < edge->dst_size(true); j++) {
        CHECK(node2idx_.find(edge->dst(j, true)) != node2idx_.end()); 
        CHECK(node2idx_.at(edge->dst(j, true)) > node2idx_.at(parent_node));
      }
    }
  }

  unordered_map<int, vector<int>> group_info;
  CHECK(group_contents_.empty());
  for (int i = 0; i < nodes_->size(); i++) {
    if (group_[i] != i) {
      group_info[FindGroup(i)].push_back(i);
    }else if (group_info.find(i) != group_info.end()){
      group_info[i].push_back(i);
    }
  }
  for (auto& iter : group_info) {
    group_contents_.push_back(std::move(iter.second)); 
  }

  return group_contents_.size();
}

void Parser::FuseGroup(int gid, list<Node*>* nodes, list<Edge*>* in_edge, list<Edge*>* out_edge) {
  //currently, we only fuse one group for accuracy testing
  CHECK(gid < group_contents_.size());
  CHECK(nodes->empty());
  CHECK(in_edge->empty());
  CHECK(out_edge->empty());
  unordered_map<Edge*, int> out_edge_times;
  auto&& ids = group_contents_[gid];
  CHECK(ids.size() > 1);
  for (int id : ids) {
    auto iter = std::next(nodes_->begin(), id);
    for (Edge* ie : (*iter)->input()) {
      if (out_edge_times.find(ie) != out_edge_times.end()) {
        out_edge_times[ie]--; 
        CHECK(out_edge_times[ie] >= 0);
        if (out_edge_times[ie] == 0) {
          out_edge_times.erase(ie); 
        }
      }else if (std::find(in_edge->begin(), in_edge->end(), ie) != in_edge->end()){
        in_edge->push_back(ie); 
      }
    }
    for (Edge* oe : (*iter)->output()) {
      CHECK(out_edge_times.find(oe) == out_edge_times.end());
      out_edge_times[oe] = oe->dst_size(); 
    }
    nodes->push_back(*std::next(nodes_->begin(), id));
  }

  for (auto iter : out_edge_times) {
    out_edge->push_back(iter.first);
  }

  {
    vector<string> output_names;
    vector<string> input_names;
    vector<TensorShapeDef> output_shapes;
    for (auto* e : *out_edge) {
      output_names.push_back(e->name()); 
      output_shapes.push_back(e->shape());
    }
    for (auto* e : *in_edge) {
      input_names.push_back(e->name()); 
    }

    OpDef op_def;
    OpDefBuilder("FusedKernel")
      .Output(output_names)
      .Input(input_names)
      .Shape(output_shapes)
      .Device("GPU")
      .Finalize(&op_def);
    Node* new_node = new SingleNode(op_def, nodes->front()->scope());
    for (auto* e : *out_edge) {
      new_node->AddOutput(e);
    }
    for (auto* e : *in_edge) {
      new_node->AddInput(e);
    }
    fused_nodes_.push_back(new_node);
  }

  remove_groups_.push_back(gid);
}

void Parser::Finalize() {
  CHECK(!remove_groups_.empty());
  for (int i = remove_groups_.size()-1; i >= 0; i--) {
    int gid = remove_groups_[i];
    int last_node_pos = group_contents_[gid][group_contents_[gid].size()-1];
    int new_node_pos = last_node_pos + 1;
    nodes_->insert(std::next(nodes_->begin(), new_node_pos), fused_nodes_[i]);
    for (int j = group_contents_[gid].size()-1; j >= 0; j--) {
      int id = group_contents_[gid][j];
      nodes_->erase(std::next(nodes_->begin(), id));
    }
  }
}

} //RTC
} //namespace midend
