#include "cavs/frontend/cxx/vertex.h"

using std::to_string;

Sym Vertex::DumpToSym() {
  Sym::SetMode(Sym::DYNAMIC_SYM);
  this->Inode();
  Sym::SetMode(Sym::STATIC_SYM);
}

Sym Vertex::GatherLocal(int i) {
  static int id = 0;
  string out = "WavefrontGatherLocal" + to_string(id++) + "_" + to_string(i);
  OpDef def = OpDefBuilder("WavefrontGatherLocal")
                .Input(vertex_data_)
                .Input(edge_data_)
                .Output(out)
                .Dtype(vertex_data_)
                .Device(vertex_data_)
                .AttrSingle("Index", i)
                .Finalize();
  return Sym(def);
}

Sym Vertex::Scatter(int i) {
  static int id = 0;
  string out = "WavefrontScatter" + to_string(id++) + "_" + to_string(i);
  OpDef def = OpDefBuilder("WavefrontScatter")
                .Input(vertex_data_)
                .Input(edge_data_)
                .Output(out)
                .Dtype(vertex_data_)
                .Device(vertex_data_)
                .AttrSingle("Index", i)
                .Finalize();
  return Sym(def);
}
