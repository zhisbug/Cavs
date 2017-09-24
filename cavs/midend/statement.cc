#include "cavs/midend/statement.h"
#include "cavs/util/timing.h"

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
  //input tensor should also be scaled in the case of the backwarding of skewed tree
  //when the forward data is reused, it's 1st dimension is the last round dim, 
  //which should be changed from round to round
  VLOG(V_TIMING) << "Setting Input/Output Tenesor Scale----";
  ctxt_->ScaleInputTensor();
  ctxt_->ScaleOutputTensor();
  //for some gradient tensor, the original value must be set to 0 
  VLOG(V_TIMING) << "Setting Zero--------------------------";
  ctxt_->SetZero();
  VLOG(V_TIMING) << "Waiting for inputs--------------------";
  ctxt_->WaitForEvent();
  VLOG(V_TIMING) << "Computing-----------------------------";
  op_->Compute(ctxt_);

  VLOG(V_TIMING) << "Recording My Event if necessary-------";
  ctxt_->RecordMyEvent();
  //checkCudaError(cudaDeviceSynchronize());
  VLOG(V_TIMING) << "======================================";
}

void GraphStatement::Run() {
  FunctionCallStatement::Run();
  CHECK(node_func_);
  CHECK(gscheduler_);

  if (push_arg_stmt_)
    push_arg_stmt_->Run();

  //checkCudaError(cudaDeviceSynchronize());
  //LOG(INFO) << "Loading graph...";
  Timing::TimingBegin("GraphParsing");
  int output_length = gscheduler_->LoadGraph(global_ctxt_->Input(0));
  //LOG(INFO) << "Load graph done...";
  CHECK(output_length > 0);
  //we must clear the dynamic size in case previous ops have changed it;
  //The only case we have to reset the dynamic size is when the previous round sets
  //the dynamic size to a size larger than the gather output capacity,
  //which can not happen
  //LOG(INFO) << "Initialzing 1st round"; 
  gscheduler_->Initialize();
  //checkCudaError(cudaDeviceSynchronize());
  //LOG(INFO) << "Initialzing 1st round done";
  Timing::TimingEnd("GraphParsing");
  int round = 0;

  Timing::TimingBegin("RNNForward");
  while (!gscheduler_->Terminate()) {
    //LOG(INFO) << "doing job_id: " << gscheduler_->GetJobId()[0];
    VLOG(V_DEBUG) << "round: " << round++
      << "\t job_counts:" << gscheduler_->GetJobId().size()
      << "\t job_id :" << gscheduler_->GetJobId()[0];
    node_func_->Run();
    gscheduler_->ActivateNext();
  }

  //we must set dynamic size for graphoutput here
  global_ctxt_->SetDynDim(output_length);
  global_ctxt_->ScaleOutputTensor();
  if (pop_ret_stmt_)
    pop_ret_stmt_->Run();
  VLOG(V_DEBUG) << "GraphOutput done";
  Timing::TimingEnd("RNNForward");
}

void GraphGradStatement::Run() {
  FunctionCallStatement::Run();
  CHECK(node_func_);
  CHECK(gscheduler_);

  if (push_arg_stmt_)
    push_arg_stmt_->Run();

  VLOG(V_DEBUG) << "here123";
  int input_length = gscheduler_->ReverseGraph();
  //global_ctxt_->SetDynDim(-1);
  gscheduler_->Initialize();
  int round = 0;

  VLOG(V_DEBUG) << "here123";
  Timing::TimingBegin("RNNBackward");
  while (!gscheduler_->Terminate()) {
    //LOG(INFO) << "doing job_id: " << gscheduler_->GetJobId()[0];
    VLOG(V_DEBUG) << "round: " << round++
      << "\t job_counts:" << gscheduler_->GetJobId().size()
      << "\t job_id :" << gscheduler_->GetJobId()[0];
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
  Timing::TimingEnd("RNNBackward");
}

} //namespace midend
