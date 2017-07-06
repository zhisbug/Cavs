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

class SingleNode;
class ScopedNode;

class Scope {
 public:
  Scope(const Scope* father, const std::string& n);

  SingleNode* AddOp(const OpDef& op_def);
  ScopedNode* AddOptimizerOp(const OpDef& op_def);
  void AddControlDependency(const OpDef& op_def);
  TensorShapeDef AddFunction(const FunctionDef& func_def);
  void GroupAllVariables(std::vector<std::string>* vars) const;

  Scope* FindChildScope(const std::string& n, bool within=false) const;
  Edge* FindEdge(const std::string& n, bool within = false) const;
  Node* FindNode(const std::string& name) const;

  void AddNode(const Node* node);
  void AddEdge(const Edge* edge);

  friend class ScopedNode;
  friend class GraphUtil;
  inline std::string name() const { return name_; }
  std::string scoped_name() const;
  void DebugSymbolTable() const;
  std::string debug_info() const;
    
 private:
  void AddGraphOpTransformation(OpDef* new_def, const OpDef& def);
  std::string name_;
  const Scope* father_;
  std::unordered_map<std::string, Scope*> children_;
  std::unordered_map<std::string, Edge*> edge_table_;
  std::unordered_map<std::string, Edge*> in_edges_;
  //std::unordered_map<std::string, Edge*> out_edges_;
  std::set<size_t> hash_nodes_;
  std::vector<Node*> typological_sorted_nodes_;
  std::unordered_map<Node*, int> node2idx_;
};

//Scope* global_scope();
Scope* main_scope();

} //namespace midend
#endif
