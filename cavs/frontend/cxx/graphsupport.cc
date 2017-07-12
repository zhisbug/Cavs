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
    size_t dim_length = 0;
    C_AddFunction(serialization.c_str(), serialization.length(),
                  &dim, &dim_length);
    CHECK(dim_length > 0);
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
    size_t dim_length = 0;
    C_AddFunction(serialization.c_str(), serialization.length(),
                  &dim, &dim_length);
    leaf_shape = vector<int>(dim, dim+dim_length);
    CHECK(dim_length > 0);
    free(dim);
  }

  CHECK(inode_shape == leaf_shape);
  CHECK(!inode_shape.empty());
  int one_node_output_size = 1;
  for (int d : inode_shape) one_node_output_size *= d;
  int max_graph_node_count = 1;
  for (int d : raw_graph_.shape(0)) max_graph_node_count *= d;

  OpDef def = OpDefBuilder("GraphOutput")
                .Input(raw_graph_.output(0))
                .Input(raw_vertex_.output(0))
                .Dtype(raw_vertex_.type())
                .Device(raw_vertex_.device())
                .Shape({-1, one_node_output_size})
                .AttrSingle("Wavefront", true)
                .AttrSingle("MaxGraphNodeCount", max_graph_node_count)
                .Finalize();
  return Sym(def);
}

Sym GraphSupport::Gather(int child,
    const std::vector<int>& shape) {
  CHECK(!shape.empty());
  OpDef def = OpDefBuilder("Gather")
                .Dtype(raw_vertex_.type())
                .Device(raw_vertex_.device())
                .Shape(shape)
                .AttrSingle("Child", child)
                //.AttrSingle("Offset", 0)
                .Finalize();
  if (__internal_unit.empty()) {
    __internal_unit = shape; 
  }else {
    CHECK(__internal_unit == shape);
  }

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
                .Shape()
                .Finalize();
  Sym AddToFunction(def);
}

void GraphSupport::Scatter(const Sym& s) {
  CHECK(!__internal_unit.empty());
  OpDef def = OpDefBuilder("Scatter")
                .Input(s.output(0))
                .Dtype(s.type())
                .Device(s.device())
                .Shape(__internal_unit) //is this really needed?
                .Finalize();
  Sym AddToFunction(def);
}
