#include "cavs/midend/op_def_builder.h"

namespace cavs {

OpDefBuilder::OpDefBuilder(string op_name) {
  op_def_.set_name(op_name);
}

OpDefBuilder& OpDefBuilder::Input(string input) {
  op_def_.add_input(input);
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(string output) {
  op_def_.add_output(output);
  return *this;
}

OpDefBuilder& OpDefBuilder::Device(string dev) {
  if (dev == "GPU")
    op_def_.set_device(GPU);
  else 
    op_def_.set_device(CPU);
  return *this;
}

OpDefBuilder& OpDefBuilder::Shape(std::initializer_list<int> shape) {
  OpDef::AttrDef* attr = op_def_.add_attr();
  attr->set_name("shape");
  OpDef::AttrType::ListValue* ishape = attr->mutable_value()->mutable_list();
  for (int dim : shape)
    ishape->add_i(dim);
  return *this;
}

void OpDefBuilder::AddToOpChainDef(OpChainDef* op_chain_def) {
  OpDef* op_def = op_chain_def->add_op();
  *op_def = op_def_;
}

void OpDefBuilder::Finalize(OpDef* op_def) {
  *op_def = op_def_;
}

} //namespace cavs
