#include "cavs/midend/runtime_compiler/parser.h"
#include "cavs/midend/node.h"
#include "cavs/util/logging.h"

#include <algorithm>
#include <map>

using std::list;
using std::vector;
using std::set;
using std::string;
using std::unordered_map;
using std::map;

namespace midend {
namespace RTC {

bool isElementwise(const string& op) {
  static vector<string> elementwise_ops =
    {"Add", "Minus", "Mul", "Tanh", "Mirror", "Sigmoid"};
  return (std::find(elementwise_ops.begin(), elementwise_ops.end(), op)
        != elementwise_ops.end());
}

bool isDeserved(const string& op) {
  static vector<string> elementwise_ops =
    {"Add", "Minus", "Mul", "Tanh", "Sigmoid"};
  return (std::find(elementwise_ops.begin(), elementwise_ops.end(), op)
        != elementwise_ops.end());
}

bool isFusable(Node* node) {
  return node->IsSingleNode() && isElementwise(node->name());
}

bool isDeserved(Node* node) {
  return node->IsSingleNode() && isDeserved(node->name());
}

Parser::Parser(list<Node*>* n) : nodes_(n) {
  CHECK(!nodes_->empty());
  group_.resize(nodes_->size(), 0);
  auto iter = nodes_->begin();
  VLOG(V_DEBUG) << "Number of nodes: " << nodes_->size();
  for (int i = 0; i < nodes_->size(); i++, iter++) {
    group_[i] = i; 
    CHECK(node2idx_.find(*iter) == node2idx_.end());
    node2idx_[*iter] = i;
    if ((*iter)->IsSingleNode()) {
      VLOG(V_DEBUG) << "ID:\t" << i;
      VLOG(V_DEBUG) << dynamic_cast<SingleNode*>(*iter)->op_def().DebugString();
    }
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
  vector<int> fusion_benefit(nodes_->size(), 0);
  for (int i = 0; i < nodes_->size(); i++, iter++) {
    if (isFusable(*iter)) {
      if (isDeserved(*iter)) fusion_benefit[i] += 1;
      CHECK((*iter)->output_size() == 1);
      Edge* edge = (*iter)->output(0);
      if (edge->dst_size() > 0) {
        Node* parent_node = edge->dst(0, true);
        CHECK(node2idx_.find(parent_node) != node2idx_.end());
        if (isFusable(parent_node)) {
          CHECK(node2idx_.at(parent_node) > i);
          group_[i] = FindGroup(node2idx_.at(parent_node));
          fusion_benefit[node2idx_.at(parent_node)] += fusion_benefit[i];
        }
        for (int j = 1; j < edge->dst_size(true); j++) {
          CHECK(node2idx_.find(edge->dst(j, true)) != node2idx_.end()); 
          CHECK(node2idx_.at(edge->dst(j, true)) > node2idx_.at(parent_node));
        }
      }
    }
  }

  map<int, vector<int>> group_info;
  CHECK(group_contents_.empty());
  for (int i = 0; i < nodes_->size(); i++) {
    if (group_[i] != i) {
      group_info[FindGroup(i)].push_back(i);
    }else if (group_info.find(i) != group_info.end()){
      group_info[i].push_back(i);
    }
  }
  for (auto& iter : group_info) {
    VLOG(V_DEBUG) << "GroupID:\t" << iter.first;
    if (fusion_benefit[iter.first] > 1) {
      group_contents_.push_back(std::move(iter.second)); 
    }
  }

  return group_contents_.size();
}

void Parser::FuseGroup(int gid, list<Node*>* nodes, list<Edge*>* in_edge, list<Edge*>* out_edge) {
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
      }else if (std::find(in_edge->begin(), in_edge->end(), ie) == in_edge->end()){
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

  CHECK(!in_edge->empty());
  CHECK(!out_edge->empty());
  remove_groups_.push_back(gid);
}

void Parser::AddFusedNode(Node* fused_node) {
  fused_nodes_.push_back(fused_node);
}

void Parser::Finalize() {
  CHECK(!(group_contents_.empty() ^ remove_groups_.empty()));
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
