#ifndef INDEXED_GRAPH_H_
#define INDEXED_GRAPH_H_

#include "cavs/midend/op_def.pb.h"
#include "cavs/frontend/dep_graph.h"

#include <vector>
#include <unordered_map>

namespace midend {

class IndexedGraph {
 public:
  IndexGraph(const DepGraph& def_grah);

 private:
  void Insert(const OpDef& op_def);
  std::vector<OpDef> ops_;
  std::unordered_map<std::string, int> out2idx;
};

FORCE_INLINE void IndexGraph::Insert(const OpDef& op_def) {
  ops_.push_back(op_def);
  const string& output = op_def.output();
  CHECK(out2idx.find(output) == out2idx.end());
  out2idx[output] = ops_.size() - 1;
}

} //namespace midend

#endif
