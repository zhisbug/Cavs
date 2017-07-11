#include "cavs/midend/op_context.h"

#include <string>

using std::string;
using std::unordered_map;

namespace midend {

unordered_map<string, void*> OpContext::repo_;
int OpContext::dyn_dim_ = -1;

void OpContext::SetTensorOffset() {
  if (gs_ && gs_->GetJobId() >= 0) {
    int job_id = gs_->GetJobId();
    for (auto& t : inputs_) {
      if (!t.SetOffsetWithId(job_id))
        VLOG(V_DEBUG) << t.name() << " must be a global tensor, "
                      << "and referenced as an input in a function";
    }
    for (auto& t : outputs_) {
      if (!t.SetOffsetWithId(job_id))
        VLOG(V_DEBUG) << t.name() << " must be a global tensor, "
                      << "and referenced as an output in a function";
    }
    
  }
}

void OpContext::ScaleTensor() {
  for (auto& t : inputs_) {
    if (t.IsDynamicSize() && t.dims(0) != dyn_dim()) {
      t.ScaleDynamicDimension(dyn_dim());
    } 
  }
  for (auto& t : outputs_) {
    if (t.IsDynamicSize() && t.dims(0) != dyn_dim()) {
      t.ScaleDynamicDimension(dyn_dim());
    } 
  }
}

void OpContext::SetZero() {
  for (auto& t : outputs_) {
    if (t.ZeroInitEnforced()) {
      t.InitWithZero(round());
    }
  }
}

string OpContext::debug_info() const {
  string info;
  for (unsigned i = 0; i < inputs_.size(); i++) {
    info += "input tensor[" + std::to_string(i)
            + "]:\t" + inputs_[i].name();
    info += "\n";
  }
  for (unsigned i = 0; i < outputs_.size(); i++) {
    info += "output tensor[" + std::to_string(i)
            + "]:\t" + outputs_[i].name();
    info += "\n";
  }
  return info;
}

} //namespace midend
