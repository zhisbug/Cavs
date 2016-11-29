#include "cavs/frontend/cxx/sym.h"
#include "cavs/core/device.pb.h"

OpDef& sym::Finalize() {
  op_def_.reset(new OpDef);
  for (auto* sym : input_)
    op_def_->add_input(sym->name());
  for (auto* sym : output_)
    op_def_->add_output(sym->name());
  op_def_->set_out_type(type);
  if (device_ == "GPU")
    op_def_->set_device(cavs::GPU);
  else
    op_def_->set_device(cavs::CPU);
  OpDef::AttrDef* attr = op_def_->set_attr();
  attr->name = "shape";
  attr->attr = shape_;
}
