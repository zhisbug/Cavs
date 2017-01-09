#ifndef CAVS_MIDEND_OP_CONTEXT_H_
#define CAVS_MIDEND_OP_CONTEXT_H_

#include "cavs/midend/tensor.h"
#include "cavs/midend/session.h"
#include "cavs/proto/op_def.pb.h"

namespace midend {

class OpContext {
 public:
  OpContext(const OpDef& op_def, SessionBase* sb);
  inline const Tensor& Input(int idx) { return *(inputs_.at(idx)); }
  inline Tensor* Output(int idx) { return outputs_.at(idx); }
 private:
  vector<const Tensor*> inputs_;
  vector<Tensor*> outputs_;
};


} //namespace midend
        
#endif
