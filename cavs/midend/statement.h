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
};

class ExprStatement : public Statement {
 public:
  ExprStatement() : op_(NULL), ctxt_(NULL) {}
  ~ExprStatement() {
    if (op_) free(op_);
    if (ctxt_) free(ctxt_);
  }
  void Run() override {
    op_->Compute(ctxt_);
  }
  inline void SetOp(OpImpl* op) { op_ = op; }

 private:
   OpImpl* op_;
   OpContext* ctxt_;
};

class BasicBlock : public Statement {
 public:
  BasicBlock() : iter_(1), stmts_(0) {}
  ~BasicBlock() {
    for (auto* stmt : stmts_)
      free(stmt);
  }
  void Run() override {
    for (int i = 0; i < iter_; i++) {
      for (auto* stmt : stmts_)  
        stmt->Run();
    }
  }
  inline void AppendStmt(Statement* stmt) {
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
