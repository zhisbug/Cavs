#ifndef CAVS_MIDEND_OP_CONTEXT_H_
#define CAVS_MIDEND_OP_CONTEXT_H_

#include "cavs/midend/tensor.h"
#include "cavs/proto/op_def.pb.h"

#include <unordered_map>
#include <string>

namespace midend {

class OpContext {
 public:
  OpContext() : round_(0), dyn_dim_(0) {}
  inline const Tensor& Input(int idx) const;
  inline Tensor* Output(int idx);
  inline int InputSize() const;
  inline int OutputSize() const;
  inline void AppendInput(const Tensor& t);
  inline void AppendOutput(const Tensor& t);

  inline void SetRound(int r) { round_ = r; }
  inline int round() const { return round_; }
  inline void ResetDynDim() { CHECK(dyn_dim_ > 0); dyn_dim_ = 0; }
  inline int dyn_dim() const { return dyn_dim_; }
  //inline void ScaleTensor(int new_dim);

  std::string debug_info() const;
  static std::unordered_map<std::string, void*> repo_;
 private:
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  int round_;
  int dyn_dim_;
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

//inline void OpContext::ScaleTensor(int new_dim) {
  //bool scaled = false;
  //for (auto& t : inputs_) {
    //if (t.IsDynamicSize()) {
      ////we assume only one input tensor size is dynamic
      //CHECK(!scaled);
      //scaled = true;
    //} 
  //}

  //bool will_scale = false;
  //for (auto& t : outputs_) {
    //if (t.IsDynamicSize()) {
      ////we assume only one output tensor size is dynamic
      //CHECK(!will_scale);
      //will_scale = true;
      //t.ScaleShape(new_dim);
    //} 
  //}

  //CHECK(!(scaled ^ will_scale));
//}

} //namespace midend
        
#endif
