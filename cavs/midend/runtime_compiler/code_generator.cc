#include "cavs/midend/runtime_compiler/code_generator.h"
#include "cavs/midend/runtime_compiler/statement_builder.h"
#include "cavs/proto/types.pb.h"

using std::string;
using std::unordered_map;
using std::list;

namespace midend {
namespace RTC {

static unordered_map<int, string> DataTypeToString =
    {{DT_FLOAT, "float"}, {DT_DOUBLE, "double"}, {DT_INT32, "int"}};

string GenDeclaration(const list<Edge*>& inputs, const list<Edge*>& outputs) {
  string source = "extern \"C\" __global__ void ";

  static int kernel_id = 0;
  const string kernel_name = "FusedKernel_" + std::to_string(kernel_id++);
  source += kernel_name;

  string output_args = "(";
  for (auto* e : outputs) {
    output_args += DataTypeToString.at(e->dtype()) + " *" + e->name() + ", ";
  }
  string input_args;
  for (auto* e : inputs) {
    input_args += "const " + DataTypeToString.at(e->dtype()) + " *" + e->name() + ", ";
  }
  input_args += "const int n_elements)\n";
  source += output_args + input_args;

  return source;
}

namespace Ewise {

string EwiseGenBodyThreadIndexing(const string& inner) {
  string idx = "const int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
  idx += "if (idx < n_elements) {\n" + inner + "\n}";
  return idx;
}

string EwiseGenBodyGetInput(const list<Edge*>& inputs) {
  string var_decl;
  for (auto* e : inputs) {
    string type = DataTypeToString.at(e->dtype());
    string var_name = CodeGenerator::PrefixedVar(e->name());
    string array_ref_name = e->name() + "[idx]";
    var_decl += type + " " + var_name + " = " + array_ref_name + ";\n";
  }
  return var_decl;
}

string EwiseGenBodyAssignOutput(const list<Edge*>& outputs) {
  string array_assign;
  for (auto* e : outputs) {
    string array_ref_name = e->name() + "[idx]";
    string var_name = CodeGenerator::PrefixedVar(e->name());
    array_assign += array_ref_name + " = " + var_name + ";\n";
  }
  return array_assign;
}

} //namespace Ewise

CodeGenerator::CodeGenerator(list<Node*>* n) : parser_(n) {
  int groups = parser_.GenerateGroup();
  list<Edge*> in_edges;
  list<Edge*> out_edges;
  list<Node*> nodes;

  for (int i = 0; i < groups; i++) {
    parser_.FuseGroup(i, &nodes, &in_edges, &out_edges);    
    string source = GenDeclaration(in_edges, out_edges);
    string func_body = Ewise::EwiseGenBodyGetInput(in_edges);
    for (auto* n : nodes) {
      func_body +=  ExprStatementBuilder().SetNode(n).toCode();
    }
    func_body += Ewise::EwiseGenBodyAssignOutput(out_edges);
    source += "{\n" + Ewise::EwiseGenBodyThreadIndexing(func_body) + "}\n";
    kernel_source_.push_back(source);
    in_edges.clear();
    out_edges.clear();
    nodes.clear();
  }
  parser_.Finalize();
}

} //namespace RTC
} //namespace midend

