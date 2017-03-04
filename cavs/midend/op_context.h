#ifndef CAVS_MIDEND_OP_CONTEXT_H_
#define CAVS_MIDEND_OP_CONTEXT_H_

#include "cavs/midend/tensor.h"
//#include "cavs/midend/session.h"
#include "cavs/proto/op_def.pb.h"

namespace midend {

class OpContext {
 public:
  //OpContext();
  inline const Tensor& Input(int idx) const {
    CHECK(idx < inputs_.size())
      << idx << "\t" << inputs_.size();
    return inputs_.at(idx); 
  }
  inline Tensor* Output(int idx) { 
    CHECK(idx < outputs_.size());
    return &(outputs_.at(idx)); 
  }
  inline void AppendInput(const Tensor& t) {inputs_.push_back(t); }
  inline void AppendOutput(const Tensor& t) { outputs_.push_back(t); }
  inline std::string DebugInfo() {
    std::string info;
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
 private:
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
};


} //namespace midend
        
#endif
