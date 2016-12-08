#include "cavs/frontend/cxx/sym.h"
#include "cavs/midend/devices.pb.h"
#include "cavs/util/logging.h"

SymBody::SymBody() : chain_(Chain::Default()) {}

void SymBody::Finalize(cavs::OpDef* op_def) const {
  op_def->set_name(op_name_);
  for (auto* sym : input_)
    op_def->add_input(sym->output_);
  op_def->add_output(output_);
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

Sym::Sym(const string& op_name, const F_Dtype type, const Shape& shape, 
         const string& output, const string& device) {
  static int id = 0;  
  body_.reset(new SymBody());
  body_->op_name_ = op_name;
  body_->type_ = type;
  body_->shape_ = shape; 
  body_->device_ = device; 
  body_->output_ = output == "" ? (op_name + std::to_string(id++)) : output;
  body_->chain_->push_back(body_.get());
}

Sym Sym::Variable(F_Dtype type, Shape shape, string output, string device) {
  return Sym("Variable", type, shape, output, device);
}

Sym Sym::Placeholder(F_Dtype type, Shape shape, string output, string device) {
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

void Sym::print() {
  //hack here
  int length = 1;
  CHECK_NOTNULL(body_.get());
  for (int dim : body_->shape_)
    length *= dim;
  if (body_->type_ == F_FLOAT) {
    for (int i = 0; i < length; i++)
      LOG(INFO) << (float)((float*)body_->raw_data)[i];
  }
}

