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

class FunctionCallStatement : public Statement {
 public:
  ~FunctionCallStatement() {
    if (push_arg_stmt_)   free(push_arg_stmt_);
    if (pop_ret_stmt_)    free(pop_ret_stmt_);
    if (global_ctxt_)     free(global_ctxt_);
  }
  inline void SetPushArgStatement(ExprStatement* push_arg) {
    push_arg_stmt_ = push_arg; 
  }
  inline void SetPopRetStatement(ExprStatement* pop_ret) {
    pop_ret_stmt_ = pop_ret; 
  }
  inline void SetGlobalContext(OpContext* ctxt) {
    global_ctxt_ = ctxt;
  }

  void Run() override {
    CHECK_NOTNULL(push_arg_stmt_);
    CHECK_NOTNULL(pop_ret_stmt_);
    CHECK_NOTNULL(global_ctxt_);
  }

 protected:
  FunctionCallStatement()
    : push_arg_stmt_(NULL), pop_ret_stmt_(NULL), global_ctxt_(NULL) {}
  ExprStatement* push_arg_stmt_;
  ExprStatement* pop_ret_stmt_;
  OpContext *global_ctxt_;
  //OpImpl *push_op_, *pop_op_;
  //OpContext *push_ctxt_, *pop_ctxt_;
};

class GraphStatement : public FunctionCallStatement {
 public:
  GraphStatement(Statement* node_func, GraphSchedulerBase* gs)
    : node_func_(node_func), gscheduler_(gs) {}
  void Run() override;

 protected:
  Statement* node_func_;
  GraphSchedulerBase* gscheduler_;
};

class GraphGradStatement : public GraphStatement {
 public:
  GraphGradStatement(Statement* node_func, GraphSchedulerBase* gs)
    : GraphStatement(node_func, gs) {}
  void Run() override;
};

} //namespace midend

#endif
