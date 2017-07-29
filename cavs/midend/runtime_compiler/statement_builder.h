#ifndef CAVS_MIDEND_RUNTIME_COMPILER_STATEMENT_BUILDER_H_
#define CAVS_MIDEND_RUNTIME_COMPILER_STATEMENT_BUILDER_H_

#include "cavs/midend/runtime_compiler/expression.h"
#include "cavs/midend/node.h"

namespace midend {
namespace RTC {

class AssignStatementBuilder {
 public:
  ~AssignStatementBuilder() { if (ae_) free (ae_); }
  AssignStatementBuilder& SetNode(Node* n);
  virtual std::string toCode() const;

 protected:
  AssignExpression* ae_;
};

class VarDeclStatementBuilder : public AssignStatementBuilder {
 public:
  std::string toCode() const override;
};

} //namespace RTC
} //namespace midend

#endif

