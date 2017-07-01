#include "cavs/midend/statement.h"

namespace midend {

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
