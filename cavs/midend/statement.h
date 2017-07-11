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
  inline static void IncRound() {
    round_++;
  }

 protected:
  inline static int round() { return round_; }

 private:
  static int round_;
};

class ExprStatement : public Statement {
 public:
  ExprStatement(OpImpl* op, OpContext* ctxt) : op_(op), ctxt_(ctxt) {}
  ~ExprStatement() {
    if (op_) free(op_);
    if (ctxt_) free(ctxt_);
  }
  inline void SetOp(OpImpl* op) { op_ = op; }
  inline void SetContext(OpContext* ctxt) { ctxt_ = ctxt; }

  void Run() override;

 protected:
  ExprStatement() : op_(NULL), ctxt_(NULL) {}
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

class GraphStatement : public ExprStatement {
 public:
  GraphStatement(Statement* leaf, Statement* inode, GraphScheduler* gs)
    : ExprStatement(), leaf_(leaf), inode_(inode), gscheduler_(gs) {}

  void Run() override;

 protected:
  Statement* leaf_;
  Statement* inode_;
  GraphScheduler* gscheduler_;
};

class GraphGradStatement : public GraphStatement {
 public:
  GraphGradStatement(Statement* leaf, Statement* inode, GraphScheduler* gs)
    : GraphStatement(leaf, inode, gs) {}

  void Run() override;
};

} //namespace midend

#endif
