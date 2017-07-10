#include "cavs/midend/op_context.h"

#include <string>

using std::string;
using std::unordered_map;

namespace midend {

unordered_map<string, void*> OpContext::repo_;
int OpContext::dyn_dim_ = -1;

void OpContext::SetTensorOffset() {
  if (gs_) {
    int job_id = -1;
    if ((job_id = gs_->GetJobId()) >= 0) {
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
}

void OpContext::ScaleTensor() {
  for (auto& t : inputs_) {
    if (t.IsDynamicSize() && t.dims(0) != dyn_dim()) {
      t.ScaleDynamicDimension(dyn_dim());
    } 
  }
  VLOG(V_DEBUG) << "here";
  for (auto& t : outputs_) {
    VLOG(V_DEBUG) << t.IsDynamicSize();
    VLOG(V_DEBUG) << t.dims(0);
    VLOG(V_DEBUG) << dyn_dim();
    if (t.IsDynamicSize() && t.dims(0) != dyn_dim()) {
      VLOG(V_DEBUG) << t.debug_info() << "\nnew_dim: " << dyn_dim();
      t.ScaleDynamicDimension(dyn_dim());
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
