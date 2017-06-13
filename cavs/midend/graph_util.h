#ifndef CAVS_MIDEND_GRAPH_UTIL_H_
#define CAVS_MIDEND_GRAPH_UTIL_H_

#include "cavs/midend/node.h"
#include "cavs/midend/scope.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/proto/func_def.pb.h"
#include "cavs/util/logging.h"

#include <vector>
#include <unordered_map>
#include <list>
#include <string>

namespace midend {

class GraphUtil {
 public:
  GraphUtil(Scope* s) : s_(s) {}
  //Node* AddNode(const OpDef& op_def);
  //const Node* FindNode(const std::string& name) const;
  //const Edge* FindEdge(const std::string& name) const;
  //void GroupAllVariables(std::vector<std::string>* vars);
  Node* AddOptimizerOp(const OpDef& op_def);
  Node* AddFunction(const FunctionDef& func_def) {
    LOG(FATAL) << "Not implemented yet";
  }
  std::string DebugInfo();

 private:
  Scope* s_;
  //bool TraverseCriticalPath(Scope*s,
      //const Edge* loss, const Edge* curr,
      //std::unordered_map<const Node*, bool>* fwd_path,
      //std::list<const Node*>* newly_traversed);
  //void GroupClosedSet(
      //const std::vector<std::string>& vars,
      //const Edge* loss,
      //const std::string& solver,
      //const float lr,
      //const float clip,
      //const std::string& proj,
      //Scope* s);
  //void DeduceAndApplyOneGradNode(
      //Scope*s,
      //const Node* node,
      //const std::string& edge);
};

} //namespace midend 

#endif
