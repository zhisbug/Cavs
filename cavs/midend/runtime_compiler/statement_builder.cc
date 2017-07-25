#include "cavs/midend/runtime_compiler/statement_builder.h"
#include "cavs/midend/runtime_compiler/code_generator.h"
#include "cavs/util/logging.h"

#include <vector>
#include <unordered_map>
#include <string>

using std::vector;
using std::unordered_map;
using std::string;

namespace midend {
namespace RTC {

ExprStatementBuilder& ExprStatementBuilder::SetNode(Node* node) {
  static unordered_map<string, string> bin_ops =
    {{"Add", "+"}, {"Minus", "-"}, {"Mul", "*"}};
  string right_hand;
  if (bin_ops.find(node->name()) != bin_ops.end()) {
    CHECK(node->input_size() == 2);
    CHECK(node->output_size() == 1);
    right_hand = BinaryExpression(bin_ops.at(node->name()),
        CodeGenerator::PrefixedVar(node->input(0)->name()),
        CodeGenerator::PrefixedVar(node->input(1)->name())).toCode();
  }else {
    LOG(FATAL) << "Wrong node";
  }
  CHECK(!ae_);
  ae_ = new AssignExpression("=",
      CodeGenerator::PrefixedVar(node->output(0)->name()), 
      right_hand);
  return *this;
}

string ExprStatementBuilder::toCode() const {
  return ExprStatement(ae_).toCode(); 
}

} //namespace RTC
} //namespace midend
