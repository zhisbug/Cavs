#include "cavs/midend/graph_session.h"

using std::string;
using std::vector;
using std::unordered_map;

namespace midend {

//For normal single node, the original scope is its running scope
//But for graph/function node, the original scope is only it defination scope
//its running scope may be belongs to the optimizer scope
string GraphSession::TensorNameInFunctionContext(const Edge* e) const {
  CHECK_NOTNULL(scope_);
  return scope_->scoped_name() + ":" + name_ + ":" + e->name();
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
    bool dynamic_size = true;
    //if (t) {
      //const_cast<Tensor*>(t)->SetZeroInitEnforced(); 
    if (!t) {
      ////all the outputs of the operators in the function are unique
      //CHECK(!t) << node->debug_info();
      const Tensor* upper_t = GetTensor(TensorNameInFunctionContext(output), true);
      //CHECK(!upper_t || (output->isGradient() && output->isVariable())) << upper_t->debug_info();
      if (upper_t) {
        VLOG(V_DEBUG) << "Found underlying tensor(" << upper_t->name()
                      << "," << upper_t->count() << " elements"
                      << ") for " << TensorNameInFunctionContext(output)
                      << " with shape info: " << output->shape().DebugString();
        VLOG(V_DEBUG) << "It must be the variable_grad(or its slice) tensor";
        Tensor out(TensorNameInFunctionContext(output), *upper_t);
        out.Reshape(output->shape());
        InsertTensor(out);
        dynamic_size = false;
      }else if (GetSingleArg<bool>(op_def, "ShareMemory", false)) {
        //currently, we only support sharing memory
        //for single-input and single-output operators
        //and only share output(0) with input(0)
        //CHECK(node->inputs_size() == 1); //reshape need two inputs
        CHECK(node->output_size() == 1); 
        CHECK_NOTNULL(GetTensor(TensorNameInFunctionContext(node->input(0)), true));
        Tensor out(TensorNameInFunctionContext(output),
            *GetTensor(TensorNameInFunctionContext(node->input(0)), true));
        out.Reshape(output->shape());
        LOG(INFO) << "[In Graph Session]: Share Memory Tensor" << out.debug_info();
        InsertTensor(out);
      }else {
        TensorShape shape;
        if (output->shape().dim_size() == 1 ||
           (output->shape().dim_size() == 2 && output->shape().dim(0) == 1)) {
          //Since the users write the think-like-a-vertex function,
          //we assume each output is one-dimension, without batching
          shape.AddDim(MAX_NODE_);
          if (output->shape().dim_size() == 1) {
            shape.AddDim(output->shape().dim(0));
          }else if (output->shape().dim_size() == 2 && output->shape().dim(0) == 1) {
            shape.AddDim(output->shape().dim(1));
          }else {
            LOG(FATAL) << "wrong dimension" << output->shape().DebugString();
          }
        }else {
          //CHECK(output->isVariable() && output->isGradient());
          //the above is wrong when deducing the backward of W.reshape.matmul 
          CHECK(output->isGradient());
          shape = TensorShape(output->shape());
          dynamic_size = false;
        }

        Allocator* alloc = GetAllocator(op_def); 
        CHECK_NOTNULL(alloc);
        VLOG(V_DEBUG) << "[In Graph Session]: Allocating full tensor for "
                      << TensorNameInFunctionContext(output)
                      << " with shape info: " << shape.debug_info();
        Tensor out(TensorNameInFunctionContext(output), alloc, op_def.dtype(), std::move(shape));
        out.Resize(output->shape());
        out.SetAsDynamic();
        VLOG(V_DEBUG) << out.debug_info();
        InsertTensor(out);
      }
      CHECK_NOTNULL(t = GetTensor(TensorNameInFunctionContext(output)));
    }
    const vector<string>& zeros =
      GetListArg<string>(dynamic_cast<const SingleNode*>(node)->op_def(), "ZeroEnforced");
    if (node->IsStatefulOp()) {
      CHECK(node->output_size() == 1);
      const_cast<Tensor*>(t)->SetZeroInitEnforced(); 
    }else if (std::find(zeros.begin(), zeros.end(), output->name()) != zeros.end()) {
      CHECK(node->name().substr(0, 11) == "FusedKernel");
      const_cast<Tensor*>(t)->SetZeroInitEnforced(); 
    }
    if (dynamic_size) const_cast<Tensor*>(t)->SetAsDynamic();

    ctxt->AppendOutput(const_cast<Tensor*>(t));
  }
  return ctxt;
}

namespace __internal {
  static unordered_map<string, GraphSession*> graph_sess_pool;
}

GraphSession* GetGraphSession(const string& name) {
  return (__internal::graph_sess_pool.find(name) == __internal::graph_sess_pool.end()) ?
          NULL : __internal::graph_sess_pool.at(name);
}

bool InsertGraphSession(const std::string& name, GraphSession* sess) {
  if (__internal::graph_sess_pool.find(name) != __internal::graph_sess_pool.end()) {
    return false;
  }else {
    __internal::graph_sess_pool.emplace(name, sess); 
    return true;
  }
}

} //namespace midend
