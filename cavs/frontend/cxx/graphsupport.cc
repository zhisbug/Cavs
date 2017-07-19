#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/op_def_builder.h"

#include <string>

using std::string;
using std::vector;

Sym GraphSupport::Output() {
  VLOG(V_DEBUG) << "Generating node functions";
  vector<int> node_shape;
  
  {
    FuncConf::FuncDefineBegin("Node");
    this->Node();
    FunctionDef func = FuncConf::FuncDefineEnd("Node");
    VLOG(V_DEBUG) << func.DebugString();
    string serialization;
    func.SerializeToString(&serialization);

    int *dim = NULL;
    size_t dim_length = 0;
    C_AddFunction(serialization.c_str(), serialization.length(),
                  &dim, &dim_length);
    CHECK(dim_length > 0);
    node_shape = vector<int>(dim, dim+dim_length);
    free(dim);
  }

  CHECK(!node_shape.empty());
  int one_node_output_size = 1;
  for (int d : node_shape) one_node_output_size *= d;
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

  VLOG(V_DEBUG) << "Generating node functions done";
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
                .Finalize();
  return Sym(def);
}

Sym GraphSupport::Pull(int offset,
    const std::vector<int>& shape) {
  OpDef def = OpDefBuilder("Pull")
                //.Input(raw_vertex_.output(0))
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
                .Finalize();
  Sym AddToFunction(def);
}

void GraphSupport::Scatter(const Sym& s) {
  OpDef def = OpDefBuilder("Scatter")
                .Input(s.output(0))
                .Dtype(s.type())
                .Device(s.device())
                .Finalize();
  Sym AddToFunction(def);
}
