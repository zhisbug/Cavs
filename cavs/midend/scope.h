#ifndef CAVS_MIDEND_SCOPE_H_
#define CAVS_MIDEND_SCOPE_H_

#include "cavs/midend/node.h"
#include "cavs/midend/edge.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/proto/func_def.pb.h"
#include "cavs/util/logging.h"

#include <string>
#include <set>
#include <list>
#include <unordered_map>

namespace midend {

class Scope {
 public:
  Scope(const Scope* father, const std::string& n);

  Node* AddOp(const OpDef& op_def);
  Node* AddOptimizerOp(const OpDef& op_def);
  TensorShapeDef AddFunction(const FunctionDef& func_def);
  void GroupAllVariables(std::vector<std::string>* vars) const;

  Scope* FindChild(const std::string& n) const;
  Edge* FindEdge(const std::string& n, bool within = false) const;
  Node* FindNode(const std::string& name) const;
  //const std::vector<Node*>& sorted_nodes() const {
    //return typological_sorted_nodes_;
  //}

  void AddNode(const Node* node);
  void AddEdge(const Edge* edge);

  friend class ScopedNode;
  friend class GraphUtil;
  void DebugSymbolTable();
  inline const std::string& name() const { return name_; }
  std::string DebugInfo();
    
 private:
  std::string name_;
  const Scope* father_;
  std::unordered_map<std::string, Scope*> children_;
  std::unordered_map<std::string, Edge*> edge_table_;
  std::unordered_map<std::string, Edge*> in_edges_;
  //std::unordered_map<std::string, Edge*> out_edges_;
  std::set<size_t> hash_nodes_;
  std::vector<Node*> typological_sorted_nodes_;
};

//Scope* global_scope();
Scope* main_scope();

} //namespace midend
#endif
