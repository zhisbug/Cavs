#include "cavs/midend/statement.h"

namespace midend {

int Statement::round_ = 0;
int Statement::dynamic_dim_ = -1;
bool Statement::dynamic_exist_ = false;

GraphStatement::GraphStatement(Statement* leaf, Statement* inode, GraphScheduler* gs)
    : ExprStatement(), leaf_(leaf), inode_(inode), gscheduler_(gs) {
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
