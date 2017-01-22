#ifndef CAVS_MIDEND_STATEMENT_H_
#define CAVS_MIDEND_STATEMENT_H_

#include <string>
#include <vector>

namespace midend {

class Statement {
 public:
  virtual void Run() {
    op_->Compute(ctxt_);
  }

 private:
   std::string name_;
   OpImpl* op_;
   OpContext* ctxt_;
};

class BasicBlock : public Statement {
 public:
  void Run() override {
    for (int i = 0; i < iter_; i++) {
      for (auto& stmt : stmts_)  
        stmt.Run();
    }
  }
  inline void Append(const Statement& stmt) {
    stmts_.push_back(stmt); 
  }

 private:
  int iter_;
  std::vector<Statement> stmts_;
};

} //namespace midend

#endif
