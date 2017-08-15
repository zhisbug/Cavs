#include "cavs/midend/op_context.h"

#include <string>

using std::string;
using std::unordered_map;

namespace midend {

unordered_map<string, void*> OpContext::repo_;
int OpContext::dyn_dim_ = -1;

void OpContext::SetTensorOffset() {
  if (gs_ && !gs_->Terminate()) {
    //input'id should be set, think about the graphoutput_grad case
    //we don't change the value or the size of input tensor buffer,
    //just choose the right offset of the input tensor buffer
    for (auto* t : inputs_) {
      if (!(t->IsFullShape())) {
        const_cast<Tensor*>(t)->SetOffsetWithId(gs_->GetCurrentRoundOffset());
      }else {
        VLOG(V_DEBUG) << t->name() << " must be a global tensor, "
                      << "and referenced as an input in a function";
      }
    }
    for (auto* t : outputs_) {
      if (!(t->IsFullShape())) {
        t->SetOffsetWithId(gs_->GetCurrentRoundOffset());
      }else {
        VLOG(V_DEBUG) << t->name() << " must be a global tensor, "
                      << "and referenced as an output in a function";
      }
    }
  }
}

void OpContext::ResetTensorOffset() {
  for (auto* t : inputs_) {
    VLOG(V_DEBUG) << "Resetting the offset of " << t->name() << "...";
    if (!(t->IsFullShape())) {
      VLOG(V_DEBUG) << "Resetted";
      const_cast<Tensor*>(t)->SetOffsetWithId(0);
    }
  }
  for (auto* t : outputs_) {
    VLOG(V_DEBUG) << "Resetting the offset of " << t->name() << "...";
    if (!(t->IsFullShape())) {
      VLOG(V_DEBUG) << "Resetted";
      t->SetOffsetWithId(0);
    }
  }
}

void OpContext::ScaleOutputTensor() {
  //Input tensor buffer size should never be modified in the operator
  for (auto* t : outputs_) {
    if (t->IsDynamicShape() && t->dims(0) != dyn_dim()) {
      VLOG(V_DEBUG) << t->name() << " [OUTPUT] first dimension change from "
                    << t->dims(0) << " to " << dyn_dim() << "\t"
                    << t << t->debug_info();
      t->ScaleDynamicDimension(dyn_dim());
    } 
  }
}

void OpContext::ScaleInputTensor() {
  for (auto* t : inputs_) {
    if (t->IsDynamicShape() && t->dims(0) != dyn_dim()) {
      //CHECK(t->dims(0) != dyn_dim()) << t->debug_info() << t;
      const_cast<Tensor*>(t)->ScaleDynamicDimension(dyn_dim());
    } 
  }
}

void OpContext::SetZero() {
  for (auto* t : outputs_) {
    if (t->ZeroInitEnforced()) {
      VLOG(V_DEBUG) << "-------------------------------------------------------";
      VLOG(V_DEBUG) << "Setting Zero for " << t->name() << " in round " << round();
      t->InitWithZero(round());
      VLOG(V_DEBUG) << "-------------------------------------------------------";
    }
  }
}

void OpContext::WaitForInputs() {
  if (stream_id_ >= 0) {
    for (int eid : inputs_event_ids_) {
      checkCudaError(cudaStreamWaitEvent(
            StreamEventHandlePool::GetCudaStream(stream_id_),
            StreamEventHandlePool::GetCudaEvent(eid), 0));
    }
  }
}

void OpContext::SyncMe() {
  if (sync_me_) {
    cudaStream_t s = (stream_id_ == -1) ?
      cudaStreamDefault : StreamEventHandlePool::GetCudaStream(stream_id_);
    checkCudaError(cudaStreamSynchronize(s));
    VLOG(V_TIMING) << "Sync!";
  }else {
    VLOG(V_TIMING) << "Not Sync!";
  }
}

string OpContext::debug_info() const {
  string info;
  for (unsigned i = 0; i < inputs_.size(); i++) {
    info += "input tensor[" + std::to_string(i)
            + "]:\t" + inputs_[i]->name();
    info += "\n";
  }
  for (unsigned i = 0; i < outputs_.size(); i++) {
    info += "output tensor[" + std::to_string(i)
            + "]:\t" + outputs_[i]->name();
    info += "\n";
  }
  return info;
}

} //namespace midend
