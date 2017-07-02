#ifndef CAVS_MIDEND_OP_CONTEXT_H_
#define CAVS_MIDEND_OP_CONTEXT_H_

#include "cavs/midend/tensor.h"
#include "cavs/proto/op_def.pb.h"

#include <unordered_map>
#include <string>

namespace midend {

class OpContext {
 public:
  OpContext() : round_(0), size_change_(false) {}
  inline const Tensor& Input(int idx) const;
  inline Tensor* Output(int idx);
  inline int InputSize() const;
  inline int OutputSize() const;
  inline void AppendInput(const Tensor& t);
  inline void AppendOutput(const Tensor& t);

  std::string debug_info() const;
  inline void SetRound(int r) { round_ = r; }
  inline int round() const { return round_; }
  inline void SetSizeFixed() { CHECK(size_changed_); size_changed_ = false; }
  inline int size_changed() const { return size_changed_; }
  inline void ScaleTensor(int new_dim);
  static std::unordered_map<std::string, void*> repo_;
 private:
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  int round_;
  int size_changed_;
};

inline const Tensor& OpContext::Input(int idx) const {
  CHECK(idx < inputs_.size())
    << idx << "\t" << inputs_.size();
  return inputs_.at(idx); 
}

inline Tensor* OpContext::Output(int idx) { 
  CHECK(idx < outputs_.size());
  return &(outputs_.at(idx)); 
}

inline int OpContext::InputSize() const {
  return inputs_.size();
}

inline int OpContext::OutputSize() const {
  return outputs_.size();
}

inline void OpContext::AppendInput(const Tensor& t) {
  inputs_.push_back(t);
}

inline void OpContext::AppendOutput(const Tensor& t) {
  outputs_.push_back(t); 
}

inline void OpContext::ScaleTensor(int new_dim) {
  bool need_scale = false;
  for (auto& t : inputs_) {
    if (t.IsDynamicSize()) {
      //we assume only one input tensor size is dynamic
      CHECK(!need_scale);
      need_scale = true;
    } 
  }

  bool will_scale = false;
  for (auto& t : outputs_) {
    if (t.IsDynamicSize()) {
      //we assume only one output tensor size is dynamic
      CHECK(!will_scale);
      will_scale = true;
      t.Scale(new_dim);
    } 
  }

  CHECK(!(need_scale ^ will_scale));
}

} //namespace midend
        
#endif
