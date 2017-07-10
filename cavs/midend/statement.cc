#include "cavs/midend/statement.h"

namespace midend {

int Statement::round_ = 0;
//int Statement::dynamic_dim_ = -1;
//bool Statement::dynamic_exist_ = false;

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
  //VLOG(V_DEBUG) << "Before ScaleTensor";
  ctxt_->ScaleTensor();
  op_->Compute(ctxt_);
  VLOG(V_TIMING) << "======================================";
}

void GraphStatement::Run() {
  CHECK(op_);
  CHECK(ctxt_);
  CHECK(leaf_);
  CHECK(inode_);
  CHECK(gscheduler_);

  int output_length = gscheduler_->LoadGraph(ctxt_->Input(0));
  CHECK(output_length > 0);
  for (int i = 0; i < gscheduler_->batch(); i++) {
    gscheduler_->ActiveLeaf(i);
    while (!gscheduler_->LeafEmpty()) {
      VLOG(V_DEBUG) << "doing leaf job_id: " << gscheduler_->GetJobId();
      //sleep(1);
      leaf_->Run();
      gscheduler_->ActiveNext();
    }
    while (!gscheduler_->InodeEmpty()) {
      VLOG(V_DEBUG) << "doing inode job_id: " << gscheduler_->GetJobId();
      //sleep(1);
      inode_->Run();
      gscheduler_->ActiveNext();
    }
  }

  ctxt_->SetDynDim(output_length);
  ExprStatement::Run();
  //ctxt_->ScaleTensor();
  //op_->Compute(ctxt_);
  VLOG(V_DEBUG) << "Graphoutput done";
}

} //namespace midend
