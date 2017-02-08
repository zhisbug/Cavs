#ifndef CAVS_MIDEND_DEP_GRAPH_H_
#define CAVS_MIDEND_DEP_GRAPH_H_

#include "cavs/midend/node.h"
#include "cavs/midend/edge.h"
#include "cavs/midend/statement.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/logging.h"

#include <vector>
#include <unordered_map>
#include <list>
#include <string>

namespace midend {

//the storage of actual data
class DepGraph {
 public:
  DepGraph(const Scope* s = GetGlobalScope())
    : s_(s) {}
  Node* AddNode(const OpDef& op_def);
  int num_nodes() const;
  const Node* operator[](int node_id) const;
  void GroupAllVariables(std::vector<std::string>* vars);
  void OptimizeWithLoss(const std::string& loss, 
      const std::string& solver, 
      const std::vector<std::string>& var_names);
  void Dump();

 private:
  const Scope* s_;
  bool TraverseCriticalPath(Scope*s,
      const Edge* loss, const Edge* curr,
      std::unordered_map<const Node*, bool>* fwd_path,
      std::list<const Node*>* newly_traversed);
  void GroupClosedSet(
      const std::vector<std::string>& vars,
      const Edge* loss,
      const std::string& solver,
      Scope* s);
};

} //namespace midend 

#endif
