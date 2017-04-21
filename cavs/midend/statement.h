#ifndef CAVS_MIDEND_STATEMENT_H_
#define CAVS_MIDEND_STATEMENT_H_

#include "cavs/midend/op_context.h"
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
  ExprStatement(OpImpl* op = NULL, OpContext* ctxt = NULL)
    : op_(op), ctxt_(ctxt) {}
  ~ExprStatement() {
    if (op_) free(op_);
    if (ctxt_) free(ctxt_);
  }
  inline void Run() override {
    CHECK(op_);
    CHECK(ctxt_);
    VLOG(V_DEBUG) << "Running Operator " << op_->DebugInfo();
    VLOG(V_DEBUG) << "Context Info \n" << ctxt_->DebugInfo();
    ctxt_->SetRound(GetRound());
    op_->Compute(ctxt_);
  }
  inline void SetOp(OpImpl* op) { op_ = op; }
  inline void SetContext(OpContext* ctxt) { ctxt_ = ctxt; }

 private:
   OpImpl* op_;
   OpContext* ctxt_;
};

class BasicBlock : public Statement {
 public:
  BasicBlock(int iter = 1)
    : iter_(iter), stmts_(0) {}
  ~BasicBlock() {
    for (auto* stmt : stmts_)
      free(stmt);
  }
  inline void Run() override {
    VLOG(V_DEBUG) << "Run " << iter_ << " iterations";
    for (int i = 0; i < iter_; i++) {
      for (auto* stmt : stmts_) {
        CHECK(stmt);
        stmt->Run();
      }
    }
  }
  inline void AppendStmt(Statement* stmt) {
    CHECK(stmt);
    stmts_.push_back(stmt); 
  }
  inline void SetStmts(const std::vector<Statement*>& stmts) {
    stmts_ = stmts;
  }
  inline void SetIter(int iter) {
    iter_ = iter; 
  }

 private:
  int iter_;
  std::vector<Statement*> stmts_;
};

} //namespace midend

#endif
