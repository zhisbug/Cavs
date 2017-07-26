#include "cavs/midend/runtime_compiler/code_generator.h"
#include "cavs/midend/runtime_compiler/statement_builder.h"
#include "cavs/util/op_def_builder.h"
#include "cavs/proto/types.pb.h"

using std::string;
using std::unordered_map;
using std::list;
using std::vector;

namespace midend {
namespace RTC {

string GenKernelName() {
  static int kernel_id = 0;
  return "FusedKernel_" + std::to_string(kernel_id++);
}

string GenKernelDeclaration(const string& kernel_name,
                            const list<Edge*>& inputs, const list<Edge*>& outputs) {
  string source = "extern \"C\" __global__ void ";

  source += kernel_name;

  string output_args = "(";
  for (auto* e : outputs) {
    output_args += CodeGenerator::typeToString(e->dtype()) + " *" + e->name() + ", ";
  }
  string input_args;
  for (auto* e : inputs) {
    input_args += "const " + CodeGenerator::typeToString(e->dtype()) + " *" + e->name() + ", ";
  }
  input_args += "const int n_elements)\n";
  source += output_args + input_args;

  return source;
}

namespace Ewise {

string EwiseGenBodyThreadIndexing(const string& inner) {
  string idx = "const int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
  idx += "if (idx < n_elements) {\n";
  idx += "printf(\"%f, %f, %f\\n\", Placeholder_0[idx], Placeholder_1[idx], Placeholder_2[idx]);\n";
  idx += inner;
  idx += "\n}";

  return idx;
}

string EwiseGenBodyGetInput(const list<Edge*>& inputs) {
  string var_decl;
  for (auto* e : inputs) {
    string type = CodeGenerator::typeToString(e->dtype());
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
  VLOG(V_DEBUG) << "here---";
  int groups = parser_.GenerateGroup();
  list<Edge*> in_edges;
  list<Edge*> out_edges;
  list<Node*> nodes;

  VLOG(V_DEBUG) << "here---";
  for (int i = 0; i < groups; i++) {
    parser_.FuseGroup(i, &nodes, &in_edges, &out_edges);    
    string name = GenKernelName();
    string source = GenKernelDeclaration(name, in_edges, out_edges);
    string func_body = Ewise::EwiseGenBodyGetInput(in_edges);
    for (auto* n : nodes) {
      func_body +=  VarDeclStatementBuilder().SetNode(n).toCode();
    }
    func_body += Ewise::EwiseGenBodyAssignOutput(out_edges);
    source += "{\n" + Ewise::EwiseGenBodyThreadIndexing(func_body) + "}\n";

    {
      vector<string> output_names;
      vector<string> input_names;
      vector<TensorShapeDef> output_shapes;
      for (auto* e : out_edges) {
        output_names.push_back(e->name()); 
        output_shapes.push_back(e->shape());
      }
      for (auto* e : in_edges) {
        input_names.push_back(e->name()); 
      }

      OpDef op_def;
      OpDefBuilder("FusedKernel")
        .Output(output_names)
        .Input(input_names)
        .Shape(output_shapes)
        .AttrSingle("KernelName", name)
        .AttrSingle("KernelSource", source)
        .Device("GPU")
        .Finalize(&op_def);
      Node* new_node = new SingleNode(op_def, nodes.front()->scope());
      for (auto* e : out_edges) {
        new_node->AddOutput(e);
      }
      for (auto* e : in_edges) {
        new_node->AddInput(e);
      }
      parser_.AddFusedNode(new_node);
    }
    kernel_source_.push_back(source);
    in_edges.clear();
    out_edges.clear();
    nodes.clear();
  }
  parser_.Finalize();
}

unordered_map<int, string> CodeGenerator::DataTypeToString =
    {{DT_FLOAT, "float"}, {DT_DOUBLE, "double"}, {DT_INT32, "int"}};

} //namespace RTC
} //namespace midend

