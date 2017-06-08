#ifndef CAVS_FRONTEND_CXX_VERTEX_H_
#define CAVS_FRONTEND_CXX_VERTEX_H_

#include "cavs/frontend/cxx/sym.h"

namespace _detail {
  class Edge {
   public:
    Edge(int offset) : offset_(offset) {}
    Sym Data(int i) {
      static int id = 0;
      string out = "Wavefront_edge_" + to_string(id++) + "_" + to_string(i);
      OpDef def = OpDefBuilder("Wavefront")
                    .Input(vertex_data_)
                    .Input(edge_data_)
                    .Output(out)
                    .Dtype(vertex_data_)
                    .Device(vertex_data_)
                    .AttrSingle("Index", i)
                    .Finalize();
    }
   private:
    int offset_;
  };
} //namespace _detail

class Graph {
 public:
  Graph(Sym edge, Sym vertex) : 
    edge_data_(edge), vertex_data_(vertex) {}
  virtual void Inode() = 0; 
  virtual void Leaf() = 0;
  Sym DumpToSym();

 protected:
  Sym GatherLocal(int i);
  _detail::Edge Gather(int i);
  Sym Scatter(int i);
  Sym Push(int i);

 private:
  Sym edge_data_;
  Sym vertex_data_;
  std::vector<_detail::Edge> in_edges_;
};

#endif
