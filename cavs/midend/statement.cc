#include "cavs/midend/statement.h"

namespace midend {

int Statement::round_ = 0;
int Statement::dynamic_dim_ = -1;
bool Statement::dynamic_exist_ = false;

void GraphStatement::Run() {
  CHECK(op_);
  CHECK(ctxt_);
  GraphScheduler::LoadGraph(ctxt_->Input(0));
  while (!GraphScheduler::LeafEmpty()) {
    leaf_->Run();
  }
  while (!GraphScheduler::InodeEmpty()) {
    inode_->Run();
  }
}

} //namespace midend
