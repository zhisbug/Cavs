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
  Node* AddOptimizerOp(const OpDef& op_def);
  void AddFunction(const FunctionDef& func_def);
  std::string DebugInfo();

 private:
  Scope* s_;
};

} //namespace midend 

#endif
