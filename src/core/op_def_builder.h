#ifndef OP_DEF_BUILDER_H_
#define OP_DEF_BUILDER_H_

#include "op_def.pb.h"
#include "graph_def.pb.h"
#include "macros.h"

#include <string>

namespace cavs{
using std::string;

class OpDefBuilder {
 public:
  explicit OpDefBuilder(string op_name);
  OpDefBuilder& Input(string input);
  OpDefBuilder& Output(string value);
  OpDefBuilder& Device(string dev);
  OpDefBuilder& Attr(string key, string value);
  string Key();
  const OpDef* op_def() const { return &op_def_; }
  void AddToGraphDef(GraphDef* graph_def);

 private:
  OpDef op_def_;
  DISALLOW_COPY_AND_ASSIGN(OpDefBuilder);
};

} //namespace cavs

#endif
