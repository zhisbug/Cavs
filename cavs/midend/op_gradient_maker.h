#ifndef CAVS_MIDEND_OP_GRADIENT_MAKER_H_
#define CAVS_MIDEND_OP_GRADIENT_MAKER_H_

#include "cavs/midend/op_def.pb.h"

namespace midend {

class OpGradientMaker {
 public:
  OpGradientMaker(const OpDef& def) : op_def_(def) {};
  virtual void MakeGradient(vector<OpDef>* def) = 0;

 protected:
  OpDef op_def_;
};

} //namespace midend

#endif
