#include "cavs/midend/graph_session.h"

using std::string;

namespace midend {

const Tensor* GraphSession::GetTensor(const string& name, bool recursive) const {
  const Tensor* t;
  if (t = SessionBase::GetTensor(name, recursive))
    return t;
  else if (t = global_sess_->GetTensor(name, recursive))
    return t;
  else 
    return NULL;
}

OpContext* GraphSession::GetContext(const Node* node) {
  //This context assign the full tensor for each operator
  //But for each function call, it may work on a specific range
  //of the whole tensor, which we will support through tensor class.
  OpContext* ctxt  = new OpContext();
  CHECK(node->IsSingleNode());
  const OpDef& op_def = dynamic_cast<const SingleNode*>(node)->op_def();
  for (auto* input : node->input()) {
    const Tensor* t = GetTensor(input->scoped_name()); 
    CHECK(t) << "Getting " << input->scoped_name();
    ctxt->AppendInput(*t);
  }

  for (auto* output : node->output()) {
    const Tensor* t = GetTensor(output->scoped_name());
    //all the outputs of the operators in the function are unique
    CHECK(!t);
    const Tensor* upper_t = GetTensor(output->scoped_name(), true);
    //all the outputs of the operators in the function are unique
    CHECK(!upper_t);
    //We assume we do not support reshape operators in the body of a function
    if (GetSingleArg<bool>(op_def, "ShareMemory", false)) {
      //currently, we only support sharing memory
      //for single-input and single-output operators
      //and only share output(0) with input(0)
      //CHECK(node->inputs_size() == 1); //reshape need two inputs
      CHECK(node->output_size() == 1); 
      Tensor out(output->scoped_name(),
          *GetTensor(node->input(0)->scoped_name()));
      out.Reshape(output->shape());
      LOG(INFO) << "[In Graph Session]: Share Memory Tensor" << out.debug_info();
      InsertTensor(out);
    }else {
      //Since the users write the think-like-a-vertex function,
      //we assume each output is one-dimension, without batching
      CHECK(output->shape().dim_size() == 1 ||
            (output->shape().dim_size() == 2 && output->shape().dim(0) == 1));
      TensorShape shape;
      shape.AddDim(MAX_NODE_);
      if (output->shape().dim_size() == 1) {
        shape.AddDim(output->shape().dim(0));
      }else if (output->shape().dim_size() == 2 && output->shape().dim(0) == 1) {
        shape.AddDim(output->shape().dim(1));
      }else {
        LOG(FATAL) << "wrong dimension" << output->shape().DebugString();
      }

      Allocator* alloc = GetAllocator(op_def); 
      CHECK_NOTNULL(alloc);
      VLOG(V_DEBUG) << "[In Graph Session]: Allocating full tensor for "
                    << output->scoped_name()
                    << " with shape info: " << shape.debug_info();
      Tensor out(output->scoped_name(), alloc, op_def.dtype(), std::move(shape));
      VLOG(V_DEBUG) << out.debug_info();
      InsertTensor(out);
    }
    t = GetTensor(output->scoped_name());
    CHECK(t) << t->debug_info();
    ctxt->AppendOutput(*t);
  }
  return ctxt;
}

} //namespace midend
