#ifndef CAVS_MIDEND_OP_CONTEXT_H_
#define CAVS_MIDEND_OP_CONTEXT_H_

#include "cavs/midend/tensor.h"
#include "cavs/midend/graph_scheduler.h"
#include "cavs/proto/op_def.pb.h"

#include <unordered_map>
#include <string>

namespace midend {

class OpContext {
 public:
  OpContext() : round_(0), gs_(NULL) {}
  inline const Tensor& Input(int idx) const;
  inline Tensor* Output(int idx);
  inline int InputSize() const;
  inline int OutputSize() const;
  inline void AppendInput(const Tensor* t);
  inline void AppendOutput(Tensor* t);

  inline void SetRound(int r) { round_ = r; }
  inline int round() const { return round_; }
  inline void SetGraphScheduler(GraphScheduler* gs) {
    CHECK(gs_ == NULL && gs);
    gs_ = gs;
  }
  inline GraphScheduler* graph_scheduler() { return gs_; }
  inline static void SetDynDim(int dyn_dim) { dyn_dim_ = dyn_dim; }

  void SetTensorOffset();
  void ScaleTensor();
  void SetZero();

  //friend class GraphGradNode;

  std::string debug_info() const;
  static std::unordered_map<std::string, void*> repo_;
 private:
  inline static int dyn_dim() { return dyn_dim_; }
  std::vector<const Tensor*> inputs_;
  std::vector<Tensor*> outputs_;
  int round_;
  GraphScheduler* gs_;
  static int dyn_dim_;
};

inline const Tensor& OpContext::Input(int idx) const {
  CHECK(idx < inputs_.size())
    << idx << "\t" << inputs_.size();
  return *(inputs_.at(idx)); 
}

inline Tensor* OpContext::Output(int idx) { 
  CHECK(idx < outputs_.size());
  return outputs_.at(idx); 
}

inline int OpContext::InputSize() const {
  return inputs_.size();
}

inline int OpContext::OutputSize() const {
  return outputs_.size();
}

inline void OpContext::AppendInput(const Tensor* t) {
  inputs_.push_back(t);
}

inline void OpContext::AppendOutput(Tensor* t) {
  outputs_.push_back(t); 
}

} //namespace midend
        
#endif
