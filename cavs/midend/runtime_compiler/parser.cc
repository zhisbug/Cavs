#include "cavs/midend/runtime_compiler/parser.h"
#include "cavs/midend/node.h"
#include "cavs/util/logging.h"

#include <algorithm>
#include <map>
#include <limits.h>

using std::list;
using std::vector;
using std::set;
using std::string;
using std::unordered_map;
using std::map;
using std::multimap;

namespace midend {
namespace RTC {

bool isElementwise(const string& op) {
  static vector<string> elementwise_ops =
    {"Add", "Minus", "Mul", "Tanh", "Mirror", "Sigmoid",
     "Assign", "Tanh_grad", "Sigmoid_grad", "Accumulate"};
  return (std::find(elementwise_ops.begin(), elementwise_ops.end(), op)
        != elementwise_ops.end());
}

bool isDeserved(const string& op) {
  static vector<string> elementwise_ops =
    {"Add", "Minus", "Mul", "Tanh", "Sigmoid",
     "Tanh_grad", "Sigmoid_grad", "Accumulate"};
  return (std::find(elementwise_ops.begin(), elementwise_ops.end(), op)
        != elementwise_ops.end());
}

bool isFusable(Node* node) {
  return node->IsSingleNode()
      && isElementwise(node->name());
      //&& false;
      //&& dynamic_cast<SingleNode*>(node)->IsBatchEnabled();
}

bool isDeserved(Node* node) {
  return node->IsSingleNode() && isDeserved(node->name());
}

Parser::Parser(list<Node*>* n, vector<vector<int>>* dependency)
  : nodes_(n), dependency_(dependency) {
  CHECK(!nodes_->empty());
  //group_.resize(nodes_->size(), 0);
  auto iter = nodes_->begin();
  VLOG(V_DEBUG) << "Number of nodes: " << nodes_->size();
  for (int i = 0; i < nodes_->size(); i++, iter++) {
    //group_[i] = i; 
    CHECK(node2idx_.find(*iter) == node2idx_.end());
    node2idx_[*iter] = i;
    if ((*iter)->IsSingleNode()) {
      VLOG(V_DEBUG) << "ID:\t" << i;
      VLOG(V_DEBUG) << dynamic_cast<SingleNode*>(*iter)->op_def().DebugString();
    }
  }
}

int FindGroup(int id, const vector<int>& group) {
  CHECK(id < group.size());
  int parent_id = group[id];
  while (parent_id != group[parent_id])
    parent_id = group[parent_id];
  return parent_id;
}

int Parser::GenerateGroup() {
  vector<int> group(nodes_->size(), 0);
  for (int i = 0; i < nodes_->size(); i++) {
    group[i] = i; 
  }
  auto iter = nodes_->begin();
  vector<int> fusion_benefit(nodes_->size(), 0);
  vector<int> bottom_line(nodes_->size(), 0);
  vector<int> top_line(nodes_->size(), INT_MAX);
  vector<bool> activated(nodes_->size(), false);
  for (int id = 0; id < nodes_->size(); id++, iter++) {
    if (isFusable(*iter)) {
      if (isDeserved(*iter)) fusion_benefit[id] += 1;
      CHECK((*iter)->output_size() == 1);
      Edge* edge = (*iter)->output(0);
      for (Node* parent_node : edge->dst(true)) {
        if (node2idx_.find(parent_node) == node2idx_.end()) continue;
        //we loose this constraint because batchweightupdater may remove some nodes in this scope
        //CHECK(node2idx_.find(parent_node) != node2idx_.end());
        if (isFusable(parent_node)) {
          int pid = node2idx_.at(parent_node);
          CHECK(pid > id);
          int gpid = FindGroup(pid, group);
          int gid = FindGroup(id, group);
          //CHECK(gpid > gid) << pid << "\t" << id << "\t" << gpid << "\t" << gid;
          if(gpid > gid){
            group[gid] = gpid;
          }else{
            group[gpid] = gid;
          }

          bottom_line[FindGroup(id, group)] = std::max(bottom_line[gid], bottom_line[pid]);
          top_line[FindGroup(id, group)] = std::min(top_line[gid], top_line[pid]);
          fusion_benefit[gpid] += fusion_benefit[gid];
          activated[pid] = true;
          if (!activated[id]) {
            bottom_line[FindGroup(id, group)] = id;
          }
        }else {
          top_line[FindGroup(id, group)] = id;
        }
      }
    }
  }

  map<int, vector<int>> group_info;
  CHECK(group_contents_.empty());
  for (int i = 0; i < nodes_->size(); i++) {
    if (group[i] != i) {
      group_info[FindGroup(i, group)].push_back(i);
    }else if (group_info.find(i) != group_info.end()){
      group_info[i].push_back(i);
    }
  }
  for (auto& iter : group_info) {
    VLOG(V_DEBUG) << "GroupID:\t" << iter.first;
    for (int id : iter.second)
      VLOG(V_DEBUG) << "GroupContent:\t" << id;
    if (fusion_benefit[iter.first] > 1) {
      group_contents_.push_back(std::move(iter.second)); 
      CHECK(bottom_line[iter.first] < top_line[iter.first]);
      group_insert_pos_.push_back(bottom_line[iter.first]);
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
      //loose this constraint because of accumulate operator
      //CHECK(out_edge_times.find(oe) == out_edge_times.end());
      out_edge_times[oe] = oe->dst_size(); 
    }
    nodes->push_back(*std::next(nodes_->begin(), id));
    auto ret = remove_nodes_.insert(*iter);
    CHECK(ret.second);
  }

  for (auto iter : out_edge_times) {
    out_edge->push_back(iter.first);
  }

  CHECK(!in_edge->empty());
  CHECK(!out_edge->empty());
}

void Parser::AddFusedNode(Node* fused_node, int gid) {
  CHECK(gid == fused_nodes_.size());
  CHECK(gid < group_contents_.size());
  fused_nodes_.push_back(fused_node);
  for (int id : group_contents_[gid]) {
    Node* n = *std::next(nodes_->begin(), id);
    CHECK(node2groupnode_.find(n) == node2groupnode_.end());
    node2groupnode_[n] = fused_node;
    groupnode2node_[fused_node].push_back(n);
  }
}

void Parser::Finalize() {
  CHECK(!(group_contents_.empty() ^ remove_nodes_.empty()));
  CHECK(group_contents_.size() == group_insert_pos_.size());
  CHECK(group_contents_.size() == fused_nodes_.size());
  multimap<int, int> pos_to_gid;
  for (int i = 0; i < group_insert_pos_.size(); i++) {
    pos_to_gid.emplace(group_insert_pos_[i], i);
  }
  for (auto riter = pos_to_gid.rbegin(); riter != pos_to_gid.rend(); riter++) {
    int pos = riter->first + 1;
    int gid = riter->second;
    nodes_->insert(std::next(nodes_->begin(), pos), fused_nodes_[gid]);
  }
  for (Node* n : remove_nodes_) {
    nodes_->remove(n);
  }

  //then we should build the new dependency
  dependency_->clear();
  dependency_->resize(nodes_->size());
  unordered_map<Node*, int> new_node2idx;
  auto iter = nodes_->begin();
  for (int i = 0; i < nodes_->size(); i++, iter++) {
    CHECK(new_node2idx.find(*iter) == new_node2idx.end());
    new_node2idx[*iter] = i; 
  }

  iter = nodes_->begin();
  for (int i = 0; i < nodes_->size(); i++, iter++) {
    if (node2idx_.find(*iter) != node2idx_.end()) {
      //not fused nodes
      for (Edge* edge : (*iter)->output()) {
        if (edge->scope() == (*iter)->scope()) {
          for (Node* pnode : edge->dst(true)) {
            if (new_node2idx.find(pnode) == new_node2idx.end()) {
              //its parent is fused nodes 
              //CHECK(node2idx_.find(pnode) != node2idx_.end());
              //we loose this constraint because batchweightupdater may remove some nodes in this scope
              if (node2idx_.find(pnode) == node2idx_.end()) continue;
              CHECK(node2groupnode_.find(pnode) != node2groupnode_.end());
              Node* fnode = node2groupnode_.at(pnode);
              CHECK(node2idx_.find(fnode) == node2idx_.end());
              CHECK(new_node2idx.find(fnode) != new_node2idx.end());
              //dependency_->at(new_node2idx.at(fnode)).push_back(i);
              dependency_->at(i).push_back(new_node2idx.at(fnode));
            }else {
              //acyclic graph
              CHECK(new_node2idx.at(pnode) > i);
              //dependency_->at(new_node2idx.at(pnode)).push_back(i);
              dependency_->at(i).push_back(new_node2idx.at(pnode));
            }
          }
        }
      }
    }else {
      //it is a newly generated fused node
      CHECK(groupnode2node_.find(*iter) != groupnode2node_.end());
      for (auto* origin_n : groupnode2node_.at(*iter)) {
        for (Edge* edge : origin_n->output()) {
          if (edge->scope() == (*iter)->scope()) {
            for (Node* pnode : edge->dst(true)) {
              if (new_node2idx.find(pnode) == new_node2idx.end()) {
                //its parent is a fused node
                //CHECK(node2idx_.find(pnode) != node2idx_.end());
                //we loose this constraint because batchweightupdater may remove some nodes in this scope
                if (node2idx_.find(pnode) == node2idx_.end()) continue;
                CHECK(node2groupnode_.find(pnode) != node2groupnode_.end());
                Node* fnode = node2groupnode_.at(pnode);
                CHECK(node2idx_.find(fnode) == node2idx_.end());
                CHECK(new_node2idx.find(fnode) != new_node2idx.end());
                if (fnode != (*iter)) {
                  CHECK(new_node2idx.at(fnode) > i);
                  //dependency_->at(new_node2idx.at(fnode)).push_back(i);
                  dependency_->at(i).push_back(new_node2idx.at(fnode));
                }
              }else {
                //acyclic graph
                CHECK(new_node2idx.at(pnode) > i);
                //dependency_->at(new_node2idx.at(pnode)).push_back(i);
                dependency_->at(i).push_back(new_node2idx.at(pnode));
              }
            }
          }
        }
      }
    }

  }
  iter = nodes_->begin();
  for (int i = 0; i < nodes_->size(); i++, iter++) {
    VLOG(V_DEBUG) << "ID:\t" << i;
    for (int d : dependency_->at(i))
      VLOG(V_DEBUG) << "\tDependency: " << d;
    VLOG(V_DEBUG) << (*iter)->debug_info();
  }
}

} //RTC
} //namespace midend
