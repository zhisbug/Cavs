#include "cavs/frontend/cxx/sym.h"
#include "cavs/midend/device.pb.h"
#include "cavs/util/logging.h"

Sym::Sym(const string& op_name, const Dtype type, const Shape& shape, 
         const string& output, const string& device) {
  static int id = 0;  
  body_.reset(new _internal::SymBody());
  body_->op_name_ = op_name;
  body_->type_ = type;
  body_->shape_ = shape; 
  body_->device_ = device; 
  body_->output_ = output == "" ? (op_name + std::to_string(id++)) : output;
}

void Sym::Finalize(cavs::OpDef* op_def) const {
  for (auto* sym : body_->input_)
    op_def->add_input(sym->output_);
  op_def->add_output(this->output());
  op_def->set_out_type(cavs::DataType((int)this->type()));
  //device
  if (this->device() == "GPU")
    op_def->set_device(cavs::GPU);
  else
    op_def->set_device(cavs::CPU);
  cavs::OpDef::AttrDef* attr = op_def->add_attr();
  //shape
  attr->set_name("shape");
  for (auto dim : this->shape())
    attr->mutable_value()->mutable_list()->add_i(dim);
}

Sym Sym::Variable(Dtype type, Shape shape, string output, string device) {
  return Sym("Variable", type, shape, output, device);
}

Sym Sym::Placeholder(Dtype type, Shape shape, string output, string device) {
  return Sym("Placeholder", type, shape, output, device);
}

Sym Sym::Abs(const Sym& a, string output, string device) {
  Sym s("Abs", a.type(), a.shape(), output, device);
  s.SetInput(a.body_.get());
  return s;
}

Sym Sym::Add(const Sym& a, const Sym& b, string output, string device) {
  CHECK(a.type() == b.type());
  Sym s("Add", a.type(), a.shape(), output, device);
  s.SetInput(a.body_.get());
  s.SetInput(b.body_.get());
  return s;
}

Sym& Sym::operator= (const Sym& sym) {
  this->body_ = sym.body_; 
  return *this;
}
