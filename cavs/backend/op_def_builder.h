#ifndef CAVS_BACKEND_OP_DEF_BUILDER_H_
#define CAVS_BACKEND_OP_DEF_BUILDER_H_

#include "cavs/proto/op_def.pb.h"
#include "cavs/util/macros.h"

#include <string>

namespace backend {

class OpDefBuilder {
 public:
  explicit OpDefBuilder(const std::string& op_name) {
    op_def_.set_name(op_name);
  }
  OpDefBuilder& Input(const std::string& input);
  OpDefBuilder& Input(const OpDef& def);
  OpDefBuilder& Output(const std::string& value);
  OpDefBuilder& Output(const OpDef& def);
  OpDefBuilder& Device(const std::string& dev);
  OpDefBuilder& Device(const OpDef& def);
  OpDefBuilder& Attr(const std::string& key, const std::string& value);
  OpDefBuilder& Attr(const OpDef& def);
  OpDefBuilder& Shape(std::initializer_list<int> shape);
  OpDefBuilder& Shape(const OpDef& def);
  OpDef* Finalize() { return &op_def_; }
  //void AddToOpChainDef(OpChainDef* op_chain_def);
  void Finalize(OpDef* op_def); 

 private:
  OpDef op_def_;
  DISALLOW_COPY_AND_ASSIGN(OpDefBuilder);
};

inline OpDefBuilder& OpDefBuilder::Input(const std::string& input) {
  op_def_.add_input(input);
  return *this;
}

inline OpDefBuilder& OpDefBuilder::Input(const OpDef& def) {
  for (auto& inp : def.input())
    op_def_.add_input(inp);
  return *this;
}

inline OpDefBuilder& OpDefBuilder::Output(const std::string& output) {
  op_def_.add_output(output);
  return *this;
}

inline OpDefBuilder& OpDefBuilder::Output(const OpDef& def) {
  for (auto& out : def.output())
    op_def_.add_output(out);
  return *this;
}

inline OpDefBuilder& OpDefBuilder::Device(const std::string& dev) {
  if (dev == "GPU")
    op_def_.set_device(GPU);
  else 
    op_def_.set_device(CPU);
  return *this;
}

inline OpDefBuilder& OpDefBuilder::Device(const OpDef& def) {
  op_def_.set_device(def.device());
  return *this;
}

inline void OpDefBuilder::Finalize(OpDef* op_def) {
  *op_def = op_def_;
}


} //namespace backend

#endif
