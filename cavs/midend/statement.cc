#include "cavs/midend/statement.h"

namespace midend {

int Statement::round_ = 0;
int Statement::dynamic_dim_ = -1;
bool Statement::dynamic_exist_ = false;

GraphStatement::GraphStatement(Statement* leaf, Statement* inode, GraphScheduler* gs)
    : ExprStatement(), leaf_(leaf), inode_(inode), gscheduler_(gs) {
}

void GraphStatement::Run() {
  //CHECK(op_);
  CHECK(ctxt_);
  CHECK(leaf_);
  CHECK(inode_);
  CHECK(gscheduler_);
  gscheduler_->LoadGraph(ctxt_->Input(0));
  while (!gscheduler_->LeafEmpty()) {
    leaf_->Run();
  }
  while (!gscheduler_->InodeEmpty()) {
    inode_->Run();
  }
}

} //namespace midend
