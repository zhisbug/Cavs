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
    const Tensor* t = GetTensor(TensorNameInFunctionContext(output));
    bool dynamic_shape = true;

    //all the tensors can be categoried into 3 classes
    //1) external tensor. For example, the placeholder/variable defined out of the node function
    //2) shared memory tensor. It means we can find its root tensor
    //3) local tensor. It can be batched and the first dimension should be modified
    //There is a corner case for the tensor of variable gradient or variable-reshape gradient
    //it is generated in this session and seems like a local tensor, but it should not be batched.
    //in the graphutil, all the edge that can be batched are marked,
    //and therefore if the forward edge can not be batched, the backward/gradient edge can not be batched
    if (!t) {
      const Tensor* upper_t = GetTensor(TensorNameInFunctionContext(output), true);
      if (upper_t) {
        VLOG(V_DEBUG) << "Found underlying tensor(" << upper_t->name()
                      << "," << upper_t->count() << " elements"
                      << ") for " << TensorNameInFunctionContext(output)
                      << " with shape info: " << output->shape().DebugString();
        VLOG(V_DEBUG) << "It must be the variable_grad(or its slice) tensor";
        Tensor out(TensorNameInFunctionContext(output), *upper_t);
        out.Reshape(output->shape());
        InsertTensor(out);
        dynamic_shape = false;
      }else if (GetSingleArg<bool>(op_def, "ShareMemory", false)) {
        //currently, we only support sharing memory
        //for single-input and single-output operators
        //and only share output(0) with input(0)
        //CHECK(node->inputs_size() == 1); //reshape need two inputs
        CHECK(node->output_size() == 1); 
        const Tensor* rt = NULL;
        CHECK_NOTNULL(rt = GetTensor(TensorNameInFunctionContext(node->input(0)), true));
        Tensor out(TensorNameInFunctionContext(output), *rt);
        dynamic_shape = rt->IsDynamicShape();
        out.Reshape(output->shape());
        if (dynamic_shape) {
          TensorShapeDef new_shape;
          if (output->shape().dim_size() == 1) {
            new_shape.add_dim(1);
            new_shape.add_dim(output->shape().dim(0));
            out.Reshape(new_shape);
          }else {
            CHECK(output->shape().dim(0) == 1);
          }
        }
        LOG(INFO) << "[In Graph Session]: Share Memory Tensor" << out.debug_info();
        InsertTensor(out);
      }else {
        if (!output->isGradient()) {
          //here, push/scatter op is specially supported when push is the grad of pull/gather
          //and the output names are random values.
          //they are not conventional gradient names, but need to be batched
          if (output->IsBatchEnabled() ||
              node->name() == "Push" || node->name() == "Scatter") {
            dynamic_shape = true;
          }else {
            dynamic_shape = false;
          }
        }else {
          const Tensor* ft = NULL;
          CHECK_NOTNULL(ft = GetTensor(GetOriginName(output->scoped_name()), true));
          dynamic_shape = ft->IsDynamicShape();
        }

        TensorShape full_shape;
        TensorShape partial_shape;
        if (dynamic_shape) {
          full_shape.AddDim(MAX_NODE_);
          if (output->shape().dim_size() == 1) {
            full_shape.AddDim(output->shape().dim(0));
          }else if (output->shape().dim(0) == 1) {
            for (int i = 1; i < output->shape().dim_size(); i++) {
              full_shape.AddDim(output->shape().dim(i));
            }
          }else {
            LOG(FATAL) << "not a think like a vertex design";
          }
          partial_shape = full_shape;
          partial_shape.SetDim(0, 1);
        }else {
          full_shape = std::move(TensorShape(output->shape()));
        }

        Allocator* alloc = GetAllocator(op_def); 
        CHECK_NOTNULL(alloc);
        VLOG(V_DEBUG) << "[In Graph Session]: Allocating full tensor for "
                      << TensorNameInFunctionContext(output)
                      << " with shape info: " << full_shape.debug_info();
        Tensor out(TensorNameInFunctionContext(output), alloc, op_def.dtype(), std::move(full_shape));
        if (dynamic_shape)  out.Resize(partial_shape);
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
      //currently, we only apply the feature to fusedkernel
      CHECK(node->name().substr(0, 11) == "FusedKernel");
      const_cast<Tensor*>(t)->SetZeroInitEnforced(); 
    }
    if (dynamic_shape) const_cast<Tensor*>(t)->SetAsDynamic();

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
