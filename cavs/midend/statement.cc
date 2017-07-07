#include "cavs/midend/statement.h"

namespace midend {

int Statement::round_ = 0;
int Statement::dynamic_dim_ = -1;
bool Statement::dynamic_exist_ = false;

GraphStatement::GraphStatement(Statement* leaf, Statement* inode)
    : ExprStatement(), leaf_(leaf), inode_(inode) {
  gs_ = new GraphScheduler();
}

void GraphStatement::Run() {
  //CHECK(op_);
  CHECK(ctxt_);
  gs_->LoadGraph(ctxt_->Input(0));
  while (!gs_->LeafEmpty()) {
    leaf_->Run();
  }
  while (!gs_->InodeEmpty()) {
    inode_->Run();
  }
}

} //namespace midend
