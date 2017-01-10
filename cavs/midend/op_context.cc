#include "cavs/midend/op_context.h"
#include "cavs/util/logging.h"

namespace midend {

OpContext::OpContext(const OpDef& op_def, SessionBase* sb) {
  for (const string& input : op_def.input()) {
    const Tensor* t = sb->GetTensor(input); 
    inputs_.push_back(t);
  }
  for (int i = 0; i < op_def.output_size(); i++) {
    const string& output = op_def.output(i);
    const Tensor* t = sb->GetTensor(output);
    if (!t) {
      TensorShape shape(op_def.shape(i)); 
      Allocator* alloc = GetAllocator(op_def); 
      CHECK_NOTNULL(alloc);
      Tensor out(output, alloc, op_def.dtype(), std::move(shape));
      sb->InsertTensor(out);
    }
    t = sb->GetTensor(output);
    CHECK_NOTNULL(t);
    outputs_.push_back(const_cast<Tensor*>(t));
  }
}

} //namespace midend
