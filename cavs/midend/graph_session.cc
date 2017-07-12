#include "cavs/midend/graph_session.h"

using std::string;

namespace midend {

//For normal single node, the original scope is its running scope
//But for graph/function node, the original scope is only it defination scope
//its running scope may be belongs to the optimizer scope
string GraphSession::TensorNameInFunctionContext(const Edge* e) const {
  CHECK_NOTNULL(running_scope_);
  return running_scope_->scoped_name() + ":" + e->name();
}

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
  CHECK(gscheduler_);
  ctxt->SetGraphScheduler(gscheduler_);
  CHECK(node->IsSingleNode());
  const OpDef& op_def = dynamic_cast<const SingleNode*>(node)->op_def();
  for (auto* input : node->input()) {
    //for the input scope, there might be two cases:
    //1) the input is in main scope and not moved into sub-scope(such as placeholder)
    //2) the input is in main scope and moved into sub-scope(such as slice)
    //for both cases, we use the recursive method to fetch tensor
    const Tensor* t = GetTensor(TensorNameInFunctionContext(input), true); 
    CHECK(t) << "Getting " << TensorNameInFunctionContext(input);
    ctxt->AppendInput(t);
  }

  for (auto* output : node->output()) {
    //Session of the function is the first to support "sharing" input model.
    //That means, one tensor is fed as the input of two different operators. 
    //This is the case of LSTM because C is both scattered and fed to compute H
    //For the backward, that means dC is calculated twice in two operators.
    //And therefore these two dCs should be accumulated.
    //So here we loose the constraint, if "t" exists, we set the tensor as
    //ZeroInitEnforced, and the computation is +=
    const Tensor* t = GetTensor(TensorNameInFunctionContext(output));
    if (t) {
      const_cast<Tensor*>(t)->SetZeroInitEnforced(); 
    }else {
      ////all the outputs of the operators in the function are unique
      //CHECK(!t) << node->debug_info();
      const Tensor* upper_t = GetTensor(TensorNameInFunctionContext(output), true);
      CHECK(!upper_t);
      //We assume we do not support reshape operators in the body of a function
      if (GetSingleArg<bool>(op_def, "ShareMemory", false)) {
        //currently, we only support sharing memory
        //for single-input and single-output operators
        //and only share output(0) with input(0)
        //CHECK(node->inputs_size() == 1); //reshape need two inputs
        CHECK(node->output_size() == 1); 
        Tensor out(TensorNameInFunctionContext(output),
            *GetTensor(TensorNameInFunctionContext(node->input(0))));
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
                      << TensorNameInFunctionContext(output)
                      << " with shape info: " << shape.debug_info();
        Tensor out(TensorNameInFunctionContext(output), alloc, op_def.dtype(), std::move(shape));
        out.Resize(output->shape());
        VLOG(V_DEBUG) << out.debug_info();
        InsertTensor(out);
      }
      CHECK_NOTNULL(t = GetTensor(TensorNameInFunctionContext(output)));
    }
    ctxt->AppendOutput(const_cast<Tensor*>(t));
  }
  return ctxt;
}

} //namespace midend
