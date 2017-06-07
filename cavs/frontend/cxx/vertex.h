#ifndef CAVS_FRONTEND_CXX_VERTEX_H_
#define CAVS_FRONTEND_CXX_VERTEX_H_

#include "cavs/frontend/cxx/sym.h"

namespace _detail {
  class Edge {
   public:
    Sym Data(int i);
  }
};

class Vertex {
 public:
  virtual void Inode() = 0; 
  virtual void Leaf() = 0;
  Sym DumpToSym();
  Sym vertex_data;
  Sym edge_data;

 protected:
  Sym InData(int i);
  Edge InEdge(int i);
};

#endif
