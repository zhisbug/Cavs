#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/c_api.h"
#include "cavs/proto/devices.pb.h"
#include "cavs/util/logging.h"

using std::vector;

void Sym::node::Finalize(OpDef* op_def) const {
  op_def->set_name(op_name_);
  for (const string& str: input_)
    op_def->add_input(str);
  for (const string& str: output_)
    op_def->add_output(str);
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
}

Sym::Sym(const string& op_name,
         const string& output, 
         const vector<string>& inputs, 
         const C_Dtype type,
         const string& device,
         const vector<int>& shape = {}) {
  static int id = 0;  
  node_.reset(new node());
  node_->op_name_ = op_name;
  node_->output_.push_back(output == "" ? 
    (op_name + std::to_string(id++)) : output);
  node_->input_ = inputs;
  node_->type_ = type;
  node_->shape_ = shape; 
  node_->device_ = device; 

  OpDef op_def;
  node_->Finalize(&op_def);
  string serial_def;
  op_def.SerializeToString(&serial_def);
  int *dim = NULL;
  size_t dim_length;
  C_AddNode(C_GetDefaultDG(),
      serial_def.c_str(), serial_def.length(),
      &dim, &dim_length);
  node_->shape_.clear();
  for (int i = 0; i < dim_length; i++)
    node_->shape_.push_back(dim[i]);
  free(dim);
}

Sym::Sym(const string& op_name, const string& input,
    const vector<Sym>& variables,
    const int iters,
    const string& projections) {
  CHECK(op_name == "Optimizer");
  char **var = (char**)malloc(variables.size()*sizeof(char*));
  for (int i = 0; i < variables.size(); i++) {
    CHECK(variables[i].node_->output_.size() == 1);
    var[i] = const_cast<char*>(variables[i].node_->output_[0].c_str());
  }
  char **grads;
  int grads_num = 0;
  C_GetGrad(C_GetDefaultDG(),
            input.c_str(), input.length(),
            var, variables.size(),
            projections.c_str(), projections.length(),
            iters,
            &grads, &grads_num);
  node_.reset(new node());
  node_->op_name_ = op_name;
  node_->shape_.clear();
  for (int i = 0; i < grads_num; i++) {
    node_->output_.emplace_back(grads[i]);
    free(grads[i]);
  }
  free(grads);
  free(var);
}

Sym Sym::Variable(C_Dtype type, vector<int> shape, string output, string device) {
  CHECK(shape.size() > 0);
  return Sym("Variable", output, {}, type, device, shape);
}

Sym Sym::Placeholder(C_Dtype type, vector<int> shape, string output, string device) {
  CHECK(shape.size() > 0);
  return Sym("Placeholder", output, {}, type, device, shape);
}

Sym Sym::Abs(const Sym& a, string output, string device) {
  //Sym s("Abs", output, {a.output()}, a.type(), device, a.shape());
  CHECK(a.node_->output_.size() == 1);
  Sym s("Abs", output, {a.node_->output_[0]},
        a.node_->type_, device);
  return s;
}

Sym Sym::Square(const Sym& a, string output, string device) {
  CHECK(a.node_->output_.size() == 1);
  Sym s("Square", output, {a.node_->output_[0]},
        a.node_->type_, device);
  return s;
}

Sym Sym::Add(const Sym& a, const Sym& b, string output, string device) {
  CHECK(a.node_->type_ == b.node_->type_);
  CHECK(a.node_->output_.size() == b.node_->output_.size() == 1);
  //Sym s("Add", output, {a.output(), b.output()}, a.type(), device, a.shape());
  Sym s("Add", output, {a.node_->output_[0], b.node_->output_[0]},
        a.node_->type_, device);
  return s;
}

Sym Sym::Sub(const Sym& a, const Sym& b, string output, string device) {
  CHECK(a.node_->type_ == b.node_->type_);
  CHECK(a.node_->output_.size() == b.node_->output_.size() == 1);
  Sym s("Sub", output, {a.node_->output_[0], b.node_->output_[0]},
        a.node_->type_, device);
  return s;
}

Sym Sym::Mul(const Sym& a, const Sym& b, string output, string device) {
  CHECK(a.node_->type_ == b.node_->type_);
  CHECK(a.node_->output_.size() == b.node_->output_.size() == 1);
  Sym s("Mul", output, {a.node_->output_[0], b.node_->output_[0]},
        a.node_->type_, device);
  return s;
}

Sym Sym::Optimizer(const Sym& a) {
  CHECK(a.node_->output_.size() == 1);
  Sym s("Optimizer", a.node_->output_[0]);
  return s;
}

Sym Sym::Optimizer(const Sym& a, vector<Sym> variables,
    int iters, const string& projections) {
  CHECK(variables.size() > 0);
  CHECK(iters > 0);
  CHECK(a.node_->output_.size() == 1);
  Sym s("Optimizer", a.node_->output_[0],
      variables, iters, projections);
  return s;
}

Sym& Sym::operator= (const Sym& sym) {
  this->node_ = sym.node_; 
  return *this;
}

void Sym::print() {
  //hack here
  int length = 1;
  CHECK_NOTNULL(node_.get());
  for (int dim : node_->shape_)
    length *= dim;
  if (node_->type_ == C_FLOAT) {
    for (int i = 0; i < length; i++)
      LOG(INFO) << (float)((float*)node_->raw_data)[i];
  }
}

void Sym::DumpGraph() {
  C_DumpGraph(C_GetDefaultDG());
}

