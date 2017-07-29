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

VarDeclStatementBuilder& VarDeclStatementBuilder::SetNode(Node* node) {
  static unordered_map<string, string> bin_ops =
    {{"Add", "+"}, {"Minus", "-"}, {"Mul", "*"}};
  static unordered_map<string, string> u_ops =
    {{"Tanh", "tanhf"}};
  static vector<string> ref_ops =
    {"Assign", "Mirror"};
  static vector<string> self_defined_ops =
    {"Sigmoid"};
  string right_hand;
  CHECK(node->IsSingleNode());
  if (bin_ops.find(node->name()) != bin_ops.end()) {
    CHECK(node->input_size() == 2);
    CHECK(node->output_size() == 1);
    right_hand = BinaryExpression(bin_ops.at(node->name()),
        CodeGenerator::PrefixedVar(node->input(0)->name()),
        CodeGenerator::PrefixedVar(node->input(1)->name()),
        dynamic_cast<SingleNode*>(node)->dtype()).toCode();
  }else if (u_ops.find(node->name()) != u_ops.end()){
    CHECK(node->input_size() == 1);
    CHECK(node->output_size() == 1);
    right_hand = UnaryExpression(u_ops.at(node->name()),
        CodeGenerator::PrefixedVar(node->input(0)->name()),
        dynamic_cast<SingleNode*>(node)->dtype()).toCode();
  }else if (std::find(ref_ops.begin(), ref_ops.end(), node->name())
            != ref_ops.end()) {
    CHECK(node->input_size() == 1);
    CHECK(node->output_size() == 1);
    right_hand = VarRefExpression(
        CodeGenerator::PrefixedVar(node->input(0)->name()),
        dynamic_cast<SingleNode*>(node)->dtype()).toCode();
  }else if (std::find(self_defined_ops.begin(), self_defined_ops.end(), node->name())
            != self_defined_ops.end()) {
    if (node->name() == "Sigmoid") {
      right_hand = SigmoidExpression(
          CodeGenerator::PrefixedVar(node->input(0)->name()),
          dynamic_cast<SingleNode*>(node)->dtype()).toCode();
    }else {
      LOG(FATAL) << "Wrong node";
    }
  }else {
    LOG(FATAL) << "Wrong node";
  }
  CHECK(!ae_);
  ae_ = new AssignExpression("=",
      CodeGenerator::PrefixedVar(node->output(0)->name()), 
      right_hand, dynamic_cast<SingleNode*>(node)->dtype());
  return *this;
}

string VarDeclStatementBuilder::toCode() const {
  return VarDeclStatement(ae_).toCode(); 
}

} //namespace RTC
} //namespace midend
