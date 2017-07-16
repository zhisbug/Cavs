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
  //for function support(the function body should get the offset of the whole buffer)
  ctxt_->SetTensorOffset();
  //for dynamic tensor size support(the tensor size may vary during iterations)
  ctxt_->ScaleTensor();
  //for some gradient tensor, the original value must be set to 0 
  ctxt_->SetZero();
  op_->Compute(ctxt_);
  VLOG(V_TIMING) << "======================================";
}

void GraphStatement::Run() {
  CHECK(op_);
  CHECK(ctxt_);
  //CHECK(leaf_);
  //CHECK(inode_);
  CHECK(node_func_);
  CHECK(gscheduler_);

  int output_length = gscheduler_->LoadGraph(ctxt_->Input(0));
  CHECK(output_length > 0);
  for (int i = 0; i < gscheduler_->batch(); i++) {
    gscheduler_->TrigerBatchId(i);
    while (!gscheduler_->empty()) {
      VLOG(V_DEBUG) << "doing job_id: " << gscheduler_->GetJobId()
                    << " in batch_id: " << i;
      //sleep(2);
      node_func_->Run();
      gscheduler_->ActivateNext();
    }
  }

  ctxt_->SetDynDim(output_length);
  ExprStatement::Run();
  VLOG(V_DEBUG) << "Graphoutput done";
}

void GraphGradStatement::Run() {
  CHECK(op_);
  CHECK(ctxt_);
  //CHECK(leaf_);
  //CHECK(inode_);
  CHECK(node_func_);
  CHECK(gscheduler_);

  ExprStatement::Run();

  gscheduler_->ReverseGraph();
  for (int i = 0; i < gscheduler_->batch(); i++) {
    gscheduler_->TrigerBatchId(i);
    while (!gscheduler_->empty()) {
      VLOG(V_DEBUG) << "doing job_id: " << gscheduler_->GetJobId()
                    << " in batch_id: " << i;
      //sleep(2);
      node_func_->Run();
      gscheduler_->ActivateNext();
    }
  }
  VLOG(V_DEBUG) << "Graphoutput done";
}

} //namespace midend
