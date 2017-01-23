#include "cavs/midend/statement_builder.h"

using std::vector;

namespace midend {

ExprStatement* BuildExprStatement(const Node* node) {
  return BuildExprStatement(node->op_def());
}

ExprStatement* BuildExprStatement(const OpDef& op_def) {
  ExprStatement *stmt = new ExprStatement();
  stmt->SetOp(CreateOp(op_def));
  return stmt;
  //OpContext* context = GetContext((*graph)[i]->op_def());
}

BasicBlock* BuildBasicBlock(const OpDef& op_def) {
  BasicBlock* bb = new BasicBlock();
  bb->AppendStmt(BuildExprStatement(op_def));
  return bb;
}

BasicBlock* BuildBasicBlock(const vector<OpDef>& op_defs) {
  BasicBlock* bb = new BasicBlock();
  for (auto& op_def : op_defs) {
    bb->AppendStmt(BuildExprStatement(op_def));
  }
  return bb;
}

BasicBlock* BuildBasicBlock(const std::vector<Statement*>& stmts) {
  BasicBlock* bb = new BasicBlock();
  bb->SetStmts(stmts);
  return bb;
}

}; //namespace midend
