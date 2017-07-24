#include "cavs/midend/runtime_compiler/statement_builder.h"
#include "cavs/util/logging.h"

#include <vector>
#include <unordered_map>
#include <string>

using std::vector;
using std::unordered_map;
using std::string;

namespace midend {
namespace RTC {

Expression* buildExpression(const string& op, const Node* node) {
  static unordered_map<string, string> bin_ops =
    {{"Add", "+"}, {"Minus", "-"}, {"Mul", "*"}};
  static unordered_map<string, string> assign_ops =
    {{"Assign", "="}};
  if (bin_ops.find(op) != bin_ops.end()) {
    CHECK(node->input_size() == 2);
    CHECK(node->output_size() == 1);
    return new BinaryExpression(op, node->input(0)->name(), node->input(1)->name());
  }else if (assign_ops.find(op) != assign_ops.end()) {
    CHECK(node->input_size() == 2);
    CHECK(node->output_size() == 1);
    return new AssignExpression(op, node->input(0)->name(), node->input(1)->name());
  }else {
    LOG(FATAL) << "Wrong node";
  }
}

ExprStatement* buildExprStatement(Expression* e) {
  CHECK(e->isAssignExpression());
  return new ExprStatement(dynamic_cast<AssignExpression*>(e));
}

VarDeclStatement* buildVarDeclStatement(DataType t, AssignExpression* ae) {
  return new VarDeclStatement(t, ae);
}

} //namespace RTC
} //namespace midend
