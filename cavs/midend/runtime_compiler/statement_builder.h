#ifndef CAVS_MIDEND_RUNTIME_COMPILER_STATEMENT_BUILDER_H_
#define CAVS_MIDEND_RUNTIME_COMPILER_STATEMENT_BUILDER_H_

#include "cavs/midend/runtime_compiler/expression.h"
#include "cavs/midend/node.h"

namespace midend {
namespace RTC {

class VarDeclStatementBuilder {
 public:
  ~VarDeclStatementBuilder() { if (ae_) free (ae_); }
  VarDeclStatementBuilder& SetNode(Node* n);
  std::string toCode() const;

 private:
  AssignExpression* ae_;
  //DISALLOW_COPY_AND_ASSIGN(ExprStatementBuilder);
};

} //namespace RTC
} //namespace midend

#endif

