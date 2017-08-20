#include "cavs/midend/statement.h"

namespace midend {

int Statement::round_ = 0;

void ExprStatement::Run() {
  CHECK(op_);
  CHECK(ctxt_);
  VLOG(V_TIMING) << "======================================";
  VLOG(V_TIMING) << "Running Operator " << op_->DebugInfo(V_TIMING);
  VLOG(V_DEBUG)  << "Running Operator " << op_->DebugInfo(V_DEBUG);
  VLOG(V_TIMING) << "--------------------------------------";
  VLOG(V_TIMING) << "Context Info \n"   << ctxt_->debug_info();
  //for data-dependent variable support(variable and placeholder should have the same batch_id)
  ctxt_->SetRound(round());
  VLOG(V_TIMING) << "Setting Round-------------------------";
  //for function support(the function body should get the offset of the whole buffer)
  VLOG(V_TIMING) << "Setting Offset------------------------";
  ctxt_->SetTensorOffset();
  //for dynamic tensor size support(the tensor size may vary during iterations)
  VLOG(V_TIMING) << "Setting Output Tenesor Scale-------------------------";
  ctxt_->ScaleOutputTensor();
  //for some gradient tensor, the original value must be set to 0 
  VLOG(V_TIMING) << "Setting Zero--------------------------";
  ctxt_->SetZero();
  //VLOG(V_TIMING) << "Customized context--------------------------";
  //if (custom_context_ != nullptr)
    //custom_context_(ctxt_);
  VLOG(V_TIMING) << "Waiting for inputs--------------------";
  ctxt_->WaitForInputs();
  VLOG(V_TIMING) << "Computing-----------------------------";
  op_->Compute(ctxt_);

  VLOG(V_TIMING) << "Sync With CPU-------------------------";
  //ctxt_->SyncMe();
  //checkCudaError(cudaDeviceSynchronize());
  VLOG(V_TIMING) << "======================================";
}

void GraphStatement::Run() {
  FunctionCallStatement::Run();
  CHECK(node_func_);
  CHECK(gscheduler_);

  if (push_arg_stmt_)
    push_arg_stmt_->Run();

  int output_length = gscheduler_->LoadGraph(global_ctxt_->Input(0));
  CHECK(output_length > 0);
  //we must clear the dynamic size in case previous ops have changed it;
  //The only case we have to reset the dynamic size is when the previous round sets
  //the dynamic size to a size larger than the gather output capacity,
  //which can not happen
  gscheduler_->Initialize();
  while (!gscheduler_->Terminate()) {
    //LOG(INFO) << "doing job_id: " << gscheduler_->GetJobId()[0];
    node_func_->Run();
    gscheduler_->ActivateNext();
  }

  //we must set dynamic size for graphoutput here
  global_ctxt_->SetDynDim(output_length);
  global_ctxt_->ScaleOutputTensor();
  if (pop_ret_stmt_)
    pop_ret_stmt_->Run();
  VLOG(V_DEBUG) << "GraphOutput done";
}

void GraphGradStatement::Run() {
  FunctionCallStatement::Run();
  CHECK(node_func_);
  CHECK(gscheduler_);

  if (push_arg_stmt_)
    push_arg_stmt_->Run();

  int input_length = gscheduler_->ReverseGraph();
  //global_ctxt_->SetDynDim(-1);
  gscheduler_->Initialize();
  while (!gscheduler_->Terminate()) {
    //LOG(INFO) << "doing job_id: " << gscheduler_->GetJobId()[0];
    node_func_->Run();
    gscheduler_->ActivateNext();
  }

  OpContext::SetDynDim(input_length);
  for (auto* stmt : batch_weight_updates_) {
    dynamic_cast<ExprStatement*>(stmt)->GetContext()->ResetTensorOffset();
    dynamic_cast<ExprStatement*>(stmt)->GetContext()->ScaleInputTensor();
    stmt->Run();
  }

  if (pop_ret_stmt_)
    pop_ret_stmt_->Run();
  VLOG(V_DEBUG) << "GraphOutputGrad done";
}

} //namespace midend
