#include "cavs/backend/op_def_builder.h"
#include "cavs/util/logging.h"

using std::string;
using std::initializer_list;

namespace backend {

OpDefBuilder& OpDefBuilder::Input(const string& input) {
  op_def_.add_input(input);
  return *this;
}

OpDefBuilder& OpDefBuilder::Input(const OpDef& def) {
  for (auto& inp : def.input())
    op_def_.add_input(inp);
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(const string& output) {
  op_def_.add_output(output);
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(const OpDef& def) {
  for (auto& out : def.output())
    op_def_.add_output(out);
  return *this;
}

OpDefBuilder& OpDefBuilder::Device(const string& dev) {
  if (dev == "GPU")
    op_def_.set_device(GPU);
  else 
    op_def_.set_device(CPU);
  return *this;
}

OpDefBuilder& OpDefBuilder::Device(const OpDef& def) {
  op_def_.set_device(def.device());
  return *this;
}

void OpDefBuilder::Finalize(OpDef* op_def) {
  *op_def = op_def_;
}

OpDefBuilder& OpDefBuilder::Shape(initializer_list<int> shape) {
  op_def_.clear_shape();
  TensorShapeDef* shape_def = op_def_.add_shape();
  for (int dim : shape)
    shape_def->add_dim(dim);
  return *this;
}

OpDefBuilder& OpDefBuilder::Shape(const TensorShapeDef& shape) {
  op_def_.clear_shape();
  *(op_def_.add_shape()) = shape;
  return *this;
}

OpDefBuilder& OpDefBuilder::Shape(const OpDef& def) {
  *(op_def_.mutable_shape()) = def.shape();
  return *this;
}

#define INSTANTIATE_SETSINGLEARG(T, fieldname)      \
  template <>                                       \
  OpDefBuilder& OpDefBuilder::Attr<T>(              \
      const string& key, T value) {                 \
    for (auto& attr : *(op_def_.mutable_attr())) {  \
      if (attr.name() == key) {                     \
        attr.mutable_value()->                      \
          mutable_list()->add_##fieldname(value);   \
          return *this;                             \
      }                                             \
    }                                               \
    OpDef::AttrDef *attr = op_def_.add_attr();      \
    attr->set_name(key);                            \
    attr->mutable_value()->                         \
      mutable_list()->add_##fieldname(value);       \
    return *this;                                   \
  }

INSTANTIATE_SETSINGLEARG(float, f)
INSTANTIATE_SETSINGLEARG(int, i)

void BuildConstantOpDef(OpDef* op_def, 
    const string& output,
    const TensorShapeDef& shape,
    float val) {
  OpDefBuilder("ConstOp").Output(output)
    .Shape(shape).Attr("init", val)
    .Device("GPU")
    .Finalize(op_def);
}

float GetConstFromConstantOp(const OpDef& def) {
  for (auto& attr : def.attr()) {
    if (attr.name() == "init") 
      return attr.value().f();
  }
  LOG(FATAL) << "init value not found";
}

} //namespace backend
