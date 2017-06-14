#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/op_def_builder.h"

Sym GraphSupport::Output() {
  FuncConf::FuncDefineBegin("Inode");
  this->Inode();
  FuncConf::FuncDefineEnd("Inode");
  FuncConf::FuncDefineBegin("Leaf");
  this->Leaf();
  FuncConf::FuncDefineEnd("Leaf");

  OpDef def = OpDefBuilder("GraphOutput")
                .Input(raw_graph_.output(0))
                .Input(raw_vertex_.output(0))
                .Dtype(raw_vertex_.type())
                .Device(raw_vertex_.device())
                .Shape({-1, count_})
                .AttrSingle("Wavefront", true)
                .Finalize();
  return Sym(def);
}

Sym GraphSupport::Gather(int child, int offset,
    const std::vector<int>& shape) {
  OpDef def = OpDefBuilder("Gather")
                .Input(raw_graph_.output(0))
                .Input(raw_vertex_.output(0))
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
                .Input(raw_graph_.output(0))
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
                .Input(raw_graph_.output(0))
                .Input(raw_vertex_.output(0))
                .Input(s.output(0))
                .Dtype(s.type())
                .Device(s.device())
                .Shape(s.shape(0))
                .Finalize();
  Sym AddToFunction(def);
  int push_unit = 1;
  for (int d : s.shape(0))
    push_unit *= d;
  CHECK(count_ == push_unit || count_ == 1);
  count_ = push_unit;
}

void GraphSupport::Scatter(const Sym& s) {
  OpDef def = OpDefBuilder("Scatter")
                .Input(raw_graph_.output(0))
                .Input(raw_vertex_.output(0))
                .Input(s.output(0))
                .Dtype(s.type())
                .Device(s.device())
                .Shape(s.shape(0))
                .Finalize();
  Sym AddToFunction(def);
}
