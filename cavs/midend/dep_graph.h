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

class DepGraph {
 public:
  DepGraph(Scope* s = GetGlobalScope())
    : s_(s) {}
  Node* AddNode(const OpDef& op_def);
  const Node* FindNode(const std::string& name) const;
  const Edge* FindEdge(const std::string& name) const;
  void GroupAllVariables(std::vector<std::string>* vars);
  void OptimizeWithLoss(const OpDef& op_def);
  std::string DebugInfo();

 private:
  Scope* s_;
  bool TraverseCriticalPath(Scope*s,
      const Edge* loss, const Edge* curr,
      std::unordered_map<const Node*, bool>* fwd_path,
      std::list<const Node*>* newly_traversed);
  void GroupClosedSet(
      const std::vector<std::string>& vars,
      const Edge* loss,
      const std::string& solver,
      const std::string& proj,
      Scope* s);
};

} //namespace midend 

#endif
