#include "cavs/midend/op_context.h"

#include <string>

using std::string;

namespace midend {

std::unordered_map<std::string, void*> OpContext::repo_;

void OpContext::SetTensorOffset() {
  if (gs_) {
    int job_id = gs_->GetJobId();
    VLOG(V_DEBUG) << "here" << job_id;
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
