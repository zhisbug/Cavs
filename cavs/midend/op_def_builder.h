#ifndef CAVS_MIDEND_OP_DEF_BUILDER_H_
#define CAVS_MIDEND_OP_DEF_BUILDER_H_

#include "cavs/midend/op_def.pb.h"
#include "cavs/util/macros.h"

#include <string>

using std::string;

namespace midend {

class OpDefBuilder {
 public:
  explicit OpDefBuilder(string op_name);
  OpDefBuilder& Input(string input);
  OpDefBuilder& Output(string value);
  OpDefBuilder& Device(string dev);
  OpDefBuilder& Shape(std::initializer_list<int> shape);
  OpDefBuilder& Attr(string key, string value);
  const OpDef* Finalize() const { return &op_def_; }
  //void AddToOpChainDef(OpChainDef* op_chain_def);
  void Finalize(OpDef* op_def); 

 private:
  OpDef op_def_;
  DISALLOW_COPY_AND_ASSIGN(OpDefBuilder);
};

} //namespace midend

#endif
