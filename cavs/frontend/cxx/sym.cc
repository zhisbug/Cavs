#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/c_api.h"
#include "cavs/proto/devices.pb.h"
#include "cavs/util/logging.h"

void SymBody::Finalize(OpDef* op_def) const {
  op_def->set_name(op_name_);
  for (const string& str: input_)
    op_def->add_input(str);
  op_def->add_output(output_);
  op_def->set_dtype(DataType((int)type_));
  //device
  if (device_ == "GPU")
    op_def->set_device(GPU);
  else
    op_def->set_device(CPU);
  //shape
  op_def->clear_shape();
  TensorShapeDef* shape_def = op_def->add_shape();
  for (auto dim : shape_)
    shape_def->add_dim(dim);
  LOG(INFO) << op_def->DebugString();
}

Sym::Sym(const string& op_name,
         const string& output, 
         const vector<string>& inputs, 
         const C_Dtype type,
         const string& device,
         const Shape& shape = {}) {
  static int id = 0;  
  body_.reset(new SymBody());
  body_->op_name_ = op_name;
  body_->output_ = output == "" ? (op_name + std::to_string(id++)) : output;
  body_->input_ = inputs;
  body_->type_ = type;
  body_->shape_ = shape; 
  body_->device_ = device; 

  OpDef op_def;
  body_->Finalize(&op_def);
  string serial_def;
  op_def.SerializeToString(&serial_def);
  int *dim;
  size_t dim_length;
  C_AddNode(C_GetDefaultDG(),
      serial_def.c_str(), serial_def.length(),
      &dim, &dim_length);
  body_->shape_.clear();
  for (int i = 0; i < dim_length; i++)
    body_->shape_.push_back(dim[i]);
  free(dim);
}

Sym Sym::Variable(C_Dtype type, Shape shape, string output, string device) {
  return Sym("Variable", output, {}, type, device, shape);
}

Sym Sym::Placeholder(C_Dtype type, Shape shape, string output, string device) {
  return Sym("Placeholder", output, {}, type, device, shape);
}

Sym Sym::Abs(const Sym& a, string output, string device) {
  //Sym s("Abs", output, {a.output()}, a.type(), device, a.shape());
  Sym s("Abs", output, {a.output()}, a.type(), device);
  return s;
}

Sym Sym::Add(const Sym& a, const Sym& b, string output, string device) {
  CHECK(a.type() == b.type());
  //Sym s("Add", output, {a.output(), b.output()}, a.type(), device, a.shape());
  Sym s("Add", output, {a.output(), b.output()}, a.type(), device);
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
  if (body_->type_ == C_FLOAT) {
    for (int i = 0; i < length; i++)
      LOG(INFO) << (float)((float*)body_->raw_data)[i];
  }
}

