#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/op_def_builder.h"

#include <string>

using std::string;
using std::vector;

Sym GraphSupport::Output() {
  VLOG(V_DEBUG) << "Generating inode and leaf functions";
  vector<int> inode_shape;
  {
    FuncConf::FuncDefineBegin("Inode");
    this->Inode();
    FunctionDef fi = FuncConf::FuncDefineEnd("Inode");
    VLOG(V_DEBUG) << fi.DebugString();
    string serialization;
    fi.SerializeToString(&serialization);

    int *dim = NULL;
    size_t dim_length;
    C_AddFunction(serialization.c_str(), serialization.length(),
                  &dim, &dim_length);
    inode_shape = vector<int>(dim, dim+dim_length);
    free(dim);
  }

  vector<int> leaf_shape;
  {
    FuncConf::FuncDefineBegin("Leaf");
    this->Leaf();
    FunctionDef fl = FuncConf::FuncDefineEnd("Leaf");
    string serialization;
    fl.SerializeToString(&serialization);

    int *dim = NULL;
    size_t dim_length;
    C_AddFunction(serialization.c_str(), serialization.length(),
                  &dim, &dim_length);
    leaf_shape = vector<int>(dim, dim+dim_length);
    free(dim);
  }

  CHECK(inode_shape == leaf_shape);
  int count = 1;
  for (int d : inode_shape) count *= d;

  OpDef def = OpDefBuilder("GraphOutput")
                .Input(raw_graph_.output(0))
                .Input(raw_vertex_.output(0))
                .Dtype(raw_vertex_.type())
                .Device(raw_vertex_.device())
                .Shape({-1, count})
                .AttrSingle("Wavefront", true)
                .Finalize();
  return Sym(def);
}

Sym GraphSupport::Gather(int child, int offset,
    const std::vector<int>& shape) {
  OpDef def = OpDefBuilder("Gather")
                .Dtype(raw_vertex_.type())
                .Device(raw_vertex_.device())
                .Shape(shape)
                .AttrSingle("Child", child)
                .AttrSingle("Offset", offset)
                .Finalize();
  return Sym(def);
}

Sym GraphSupport::Pull(int offset,
    const std::vector<int>& shape) {
  OpDef def = OpDefBuilder("Pull")
                .Input(raw_vertex_.output(0))
                .Dtype(raw_vertex_.type())
                .Device(raw_vertex_.device())
                .Shape(shape)
                .AttrSingle("Offset", offset)
                .Finalize();
  return Sym(def);
}

void GraphSupport::Push(const Sym& s) {
  OpDef def = OpDefBuilder("Push")
                .Input(s.output(0))
                .Dtype(s.type())
                .Device(s.device())
                //.Shape(s.shape(0))
                .Finalize();
  Sym AddToFunction(def);
}

void GraphSupport::Scatter(const Sym& s) {
  OpDef def = OpDefBuilder("Scatter")
                .Input(s.output(0))
                .Dtype(s.type())
                .Device(s.device())
                //.Shape(s.shape(0))
                .Finalize();
  Sym AddToFunction(def);
}
