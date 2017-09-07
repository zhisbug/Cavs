#ifndef CAVS_MIDEND_RUNTIME_COMPILER_PARSER_H_
#define CAVS_MIDEND_RUNTIME_COMPILER_PARSER_H_

#include "cavs/midend/node.h"

#include <list>
#include <vector>
#include <string>
#include <utility>
#include <set>
#include <unordered_map>

namespace midend {
namespace RTC {

class Parser {
 public:
  //Parser(std::list<Node*>* n, std::vector<std::vector<int>>* dependency);
  Parser(std::list<Node*>* n);
  int GenerateGroup();
  void FuseGroup(int gid, std::list<Node*>* nodes,
                 std::list<Edge*>* in_edges, std::list<Edge*>* out_Edges);
  void AddFusedNode(Node* fused_node, int gid);
  void Finalize();

 private:
  //int FindGroup(int id) const;
  std::list<Node*>* nodes_;
  //std::vector<std::vector<int>>* dependency_;

  std::unordered_map<Node*, int> node2idx_;
  std::vector<std::vector<int>> group_contents_;
  std::unordered_map<Node*, Node*> node2groupnode_;
  std::unordered_map<Node*, std::vector<Node*>> groupnode2node_;

  std::vector<int> group_insert_pos_;
  std::vector<Node*> fused_nodes_;
  std::set<Node*> remove_nodes_;
};

} //namespace RTC
} //namespace midend

#endif
