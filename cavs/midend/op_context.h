#ifndef CAVS_MIDEND_OP_CONTEXT_H_
#define CAVS_MIDEND_OP_CONTEXT_H_

#include "cavs/midend/tensor.h"
//#include "cavs/midend/session.h"
#include "cavs/proto/op_def.pb.h"

namespace midend {

class OpContext {
 public:
  //OpContext();
  inline const Tensor& Input(int idx) const { return inputs_.at(idx); }
  inline Tensor* Output(int idx) { return &(outputs_.at(idx)); }
  inline void AppendInput(const Tensor& t) { inputs_.push_back(t); }
  inline void AppendOutput(const Tensor& t) { outputs_.push_back(t); }
 private:
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
};


} //namespace midend
        
#endif
