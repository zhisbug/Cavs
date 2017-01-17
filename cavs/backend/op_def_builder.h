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
  OpDefBuilder& Attr(const std::string& key, float value);
  OpDefBuilder& Attr(const OpDef& def);
  OpDefBuilder& Shape(std::initializer_list<int> shape);
  OpDefBuilder& Shape(const TensorShapeDef& shape);
  OpDefBuilder& Shape(const OpDef& def);
  OpDef* Finalize() { return &op_def_; }
  void Finalize(OpDef* op_def); 

 private:
  OpDef op_def_;
  DISALLOW_COPY_AND_ASSIGN(OpDefBuilder);
};

void BuildConstantOpDef(OpDef* op_def, 
    const std::string& output,
    const TensorShapeDef& shape,
    float val = 1.f);

float GetConstFromConstantOp(const OpDef& def);

} //namespace backend

#endif
