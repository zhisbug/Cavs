#ifndef CAVS_MIDEND_RUNTIME_COMPILER_STATEMENT_BUILDER_H_
#define CAVS_MIDEND_RUNTIME_COMPILER_STATEMENT_BUILDER_H_

#include "cavs/midend/runtime_compiler/expression.h"
#include "cavs/midend/node.h"

namespace midend {
namespace RTC {

Expression* buildExpression(const std::string& op, const Node* node);

ExprStatement* buildExprStatement(Expression* e);
ExprStatement* buildVarDeclStatement(AssignExpression* ae);

} //namespace RTC
} //namespace midend

#endif

