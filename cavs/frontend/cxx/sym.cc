#include "cavs/frontend/cxx/sym.h"
#include "cavs/core/device.pb.h"
#include "cavs/core/logging.h"

Sym::Sym(string name, Dtype type, Shape shape, string device) 
  : name_(name), type_(type), shape_(shape), device_(device) {
  static int id = 0;  
  id_ = id++;
}
void Sym::Finalize(cavs::OpDef* op_def) const {
  for (auto* sym : input_)
    op_def->add_input(sym->name());
  //op_def->add_output(sym->name());
  op_def->add_output(name() + std::to_string(id_));
  op_def->set_out_type(cavs::DataType((int)type_));
  //device
  if (device_ == "GPU")
    op_def->set_device(cavs::GPU);
  else
    op_def->set_device(cavs::CPU);
  cavs::OpDef::AttrDef* attr = op_def->add_attr();
  //shape
  attr->set_name("shape");
  for (auto dim : shape_)
    attr->mutable_value()->mutable_list()->add_i(dim);
}

Sym* Sym::Variable(Dtype type, Shape shape, string device) {
  return new Sym("Variable", type, shape, device);
}

Sym* Sym::Placeholder(Dtype type, Shape shape, string device) {
  return new Sym("Placeholder", type, shape, device);
}

Sym* Sym::Abs(Sym* a, string device) {
  return new Sym("Abs", a->type(), a->shape(), device);
}

Sym* Sym::Add(Sym* a, Sym* b, string device) {
  CHECK(a->type() == b->type());
  return new Sym("Add", a->type(), a->shape(), device);
}
