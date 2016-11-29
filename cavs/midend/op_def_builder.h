#ifndef CAVS_MIDEND_OP_DEF_BUILDER_H_
#define CAVS_MIDEND_OP_DEF_BUILDER_H_

#include "cavs/midend/op_def.pb.h"
#include "cavs/midend/op_chain_def.pb.h"
#include "cavs/midend/macros.h"

#include <string>

using std::string;

namespace cavs{

class OpDefBuilder {
 public:
  explicit OpDefBuilder(string op_name);
  OpDefBuilder& Input(string input);
  OpDefBuilder& Output(string value);
  OpDefBuilder& Device(string dev);
  OpDefBuilder& Attr(string key, string value);
  const OpDef* Finalize() const { return &op_def_; }
  void AddToOpChainDef(OpChainDef* op_chain_def);
  void Finalize(OpDef* op_def); 

 private:
  OpDef op_def_;
  DISALLOW_COPY_AND_ASSIGN(OpDefBuilder);
};

} //namespace cavs

#endif
