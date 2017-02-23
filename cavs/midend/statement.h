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
  ExprStatement(OpImpl* op = NULL, OpContext* ctxt = NULL)
    : op_(op), ctxt_(ctxt) {}
  ~ExprStatement() {
    if (op_) free(op_);
    if (ctxt_) free(ctxt_);
  }
  inline void Run() override {
    //LOG(INFO) << "Running Operator " << op_->DebugInfo();
    CHECK(op_);
    CHECK(ctxt_);
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
    //LOG(INFO) << "Run " << iter_ << " iterations";
    for (int i = 0; i < iter_; i++) {
      int counter = 0;
      for (auto* stmt : stmts_) {
        //LOG(INFO) << "Running" << i << "\t" << counter++;
        stmt->Run();
      }
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
