#ifndef CAVS_MIDEND_RUNTIME_COMPILER_PARSER_H_
#define CAVS_MIDEND_RUNTIME_COMPILER_PARSER_H_

#include "cavs/midend/node.h"
#include "cavs/midend/runtime_compiler/expression.h"

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
  Parser(std::list<Node*>* n);
  int GenerateGroup();
  void FuseGroup(int gid, std::list<Node*>* nodes, std::list<Edge*>* in_edges, std::list<Edge*>* out_Edges);
  void Finalize();


 private:
  int FindGroup(int id) const;
  std::list<Node*>* nodes_;
  std::unordered_map<Node*, int> node2idx_;
  std::vector<int> group_;
  std::vector<std::vector<int>> group_contents_;
  std::set<int> remove_ids_;
};

} //namespace RTC
} //namespace midend

#endif
