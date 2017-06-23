#ifndef CAVS_MIDEND_STATEMENT_H_
#define CAVS_MIDEND_STATEMENT_H_

#include "cavs/midend/op_context.h"
#include "cavs/midend/graph_scheduler.h"
#include "cavs/backend/op_impl.h"

#include <string>
#include <vector>

namespace midend {

using ::backend::OpImpl;
using ::backend::CreateOp;

class Statement {
 public:
  virtual void Run() = 0;
  inline void SetRound(int r) { round_ = r; }

 protected:
  inline int GetRound() const { return round_; }

 private:
  int round_;
};

class ExprStatement : public Statement {
 public:
  ExprStatement(OpImpl* op, OpContext* ctxt) : op_(op), ctxt_(ctxt) {}
  ~ExprStatement() {
    if (op_) free(op_);
    if (ctxt_) free(ctxt_);
  }
  inline void Run() override {
    CHECK(op_);
    CHECK(ctxt_);
    VLOG(V_TIMING) << "======================================";
    VLOG(V_TIMING) << "Running Operator " << op_->DebugInfo(V_TIMING);
    VLOG(V_DEBUG)  << "Running Operator " << op_->DebugInfo(V_DEBUG);
    VLOG(V_TIMING) << "--------------------------------------";
    VLOG(V_TIMING) << "Context Info \n" << ctxt_->DebugInfo();
    ctxt_->SetRound(GetRound());
    op_->Compute(ctxt_);
    VLOG(V_TIMING) << "======================================";
  }
  inline void SetOp(OpImpl* op) { op_ = op; }
  inline void SetContext(OpContext* ctxt) { ctxt_ = ctxt; }

 private:
   OpImpl* op_;
   OpContext* ctxt_;
};

class BasicBlock : public Statement {
 public:
  BasicBlock(int iter) : iter_(iter), stmts_(0) {
    CHECK(iter > 0) ;
  }

  ~BasicBlock() {
    for (auto* stmt : stmts_)
      free(stmt);
  }

  inline void Run() override {
    VLOG(V_DEBUG) << "This Basic Block will Run " << iter_ << " iterations ";
    for (int i = 0; i < iter_; i++) {
      for (auto* stmt : stmts_) {
        stmt->Run();
      }
    }
  }
  inline Statement* AppendStmt(Statement* stmt) {
    CHECK(stmt);
    stmts_.push_back(stmt); 
    return stmt;
  }

 protected:
  int iter_;
  std::vector<Statement*> stmts_;
};

class GraphStatement : public BasicBlock {
 public:
  GraphStatement(Statement* leaf, Statement* inode)
      : BasicBlock(1) {
    leaf_ = AppendStmt(leaf);
    inode_ = AppendStmt(inode);
  }

  inline void Run() override {
    while (!GraphScheduler::LeafEmpty()) {
      leaf_->Run(); 
    }
    while (!GraphScheduler::InodeEmpty()) {
      inode_->Run(); 
    }
  }

 private:
  Statement* leaf_;
  Statement* inode_;
};

} //namespace midend

#endif
