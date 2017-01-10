#include "cavs/backend/op_def_builder.h"
#include "cavs/util/logging.h"

namespace backend {

OpDefBuilder& OpDefBuilder::Shape(std::initializer_list<int> shape) {
  op_def_.clear_shape();
  TensorShapeDef* shape_def = op_def_.add_shape();
  for (int dim : shape)
    shape_def->add_dim(dim);
  return *this;
}

OpDefBuilder& OpDefBuilder::Shape(const OpDef& def) {
  *(op_def_.mutable_shape()) = def.shape();
  return *this;
}

} //namespace midend
