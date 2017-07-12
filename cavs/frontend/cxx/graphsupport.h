#ifndef CAVS_FRONTEND_CXX_VERTEX_H_
#define CAVS_FRONTEND_CXX_VERTEX_H_

#include "cavs/frontend/cxx/sym.h"

#include <vector>

class GraphSupport {
 public:
  GraphSupport(const Sym& graph_ph, const Sym& vertex_ph) : 
    raw_graph_(graph_ph), raw_vertex_(vertex_ph),
    __internal_unit(0) {}
  //virtual void Inode() = 0; 
  //virtual void Leaf() = 0;
  virtual void Node() = 0;
  Sym Output();

 protected:
  Sym Gather(int child, const std::vector<int>& shape);
  Sym Pull(int offset, const std::vector<int>& shape);
  void Push(const Sym& s);
  void Scatter(const Sym& s);

 private:
  Sym raw_graph_;
  Sym raw_vertex_;
  std::vector<int> __internal_unit;
};

#endif
