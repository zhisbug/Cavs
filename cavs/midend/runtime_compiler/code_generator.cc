#include "cavs/midend/runtime_compiler/code_generator.h"
#include "cavs/midend/runtime_compiler/statement_builder.h"
#include "cavs/util/op_def_builder.h"
#include "cavs/proto/types.pb.h"

using std::string;
using std::unordered_map;
using std::list;
using std::vector;
using std::set;

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

  //potential bug when the graph is cyclic
  //one argument and one const argument may be the same
  //it will cause compile error in cudaRTC compilation
  string output_args = "(";
  for (auto* e : outputs) {
    output_args += CodeGenerator::typeToString(e->dtype()) + " *" + e->name() + ", ";
  }
  string input_args;
  for (auto* e : inputs) {
    input_args += "const " + CodeGenerator::typeToString(e->dtype()) + " *" + e->name() + ", ";
  }
  string output_count;
  for (auto* e : outputs) {
    output_count += "const int " + CodeGenerator::arrSize(e->name()) + ", ";
  }
  string input_count;
  for (auto* e : inputs) {
    input_count += "const int " + CodeGenerator::arrSize(e->name()) + ", ";
  }
  string total_count;
  total_count = "const int n_elements)\n";

  source += output_args + input_args + output_count + input_count + total_count;

  return source;
}

namespace Ewise {

string EwiseGenBodyThreadIndexing(const string& inner) {
  string idx = "const int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
  idx += "if (idx < n_elements) {\n";
  //idx += "printf(\"%f, %f, %f\\n\", Placeholder_0[idx], Placeholder_1[idx], Placeholder_2[idx]);\n";
  idx += inner;
  idx += "\n}";

  return idx;
}

string EwiseGenBodyGetInput(const list<Edge*>& inputs, bool bcast = true) {
  string var_decl;
  for (auto* e : inputs) {
    string type = CodeGenerator::typeToString(e->dtype());
    string var_name = CodeGenerator::PrefixedVar(e->name());
    string array_ref_name = e->name() + "[idx%" + CodeGenerator::arrSize(e->name()) + "]";
    if (!bcast)
      array_ref_name = "(idx >= " + CodeGenerator::arrSize(e->name()) + ")? 0 : " + array_ref_name;
    var_decl += type + " " + var_name + " = " + array_ref_name + ";\n";
  }
  return var_decl;
}

string EwiseGenBodyGetInput(const string& name, float init) {
  string var_decl;
  string type = CodeGenerator::typeToString(DT_FLOAT);
  string var_name = CodeGenerator::PrefixedVar(name);
  var_decl += type + " " + var_name + " = " + std::to_string(init) + ";\n";
  return var_decl;
}

string EwiseGenBodyAssignOutput(const list<Edge*>& outputs) {
  string array_assign;
  for (auto* e : outputs) {
    string array_ref_name = e->name() + "[idx%" + CodeGenerator::arrSize(e->name()) + "]";
    string var_name = CodeGenerator::PrefixedVar(e->name());
    string assignment = array_ref_name + " = " + var_name + ";\n";
    string atomic_assignment = "atomicAdd(&" + array_ref_name + ", " + var_name + ");\n";
    string branch = "if (" + CodeGenerator::arrSize(e->name()) + " < n_elements) {\n"
                  + atomic_assignment + "}else {\n" + assignment + "}\n";
    array_assign += branch;
  }
  return array_assign;
}

} //namespace Ewise

CodeGenerator::CodeGenerator(list<Node*>* n, vector<vector<int>>* dependency)
  : parser_(n, dependency) {
  int groups = parser_.GenerateGroup();
  list<Edge*> in_edges;
  list<Edge*> out_edges;
  list<Node*> nodes;

  VLOG(V_DEBUG) << groups << " Groups Found";
  for (int i = 0; i < groups; i++) {
    parser_.FuseGroup(i, &nodes, &in_edges, &out_edges);
    string name = GenKernelName();
    string source = GenKernelDeclaration(name, in_edges, out_edges);
    string func_body = Ewise::EwiseGenBodyGetInput(in_edges);
    vector<string> stateful_output;
    for (auto* n : nodes) {
      VLOG(V_DEBUG) << dynamic_cast<SingleNode*>(n)->op_def().DebugString();
      if (n->IsStatefulOp() &&
          std::find(stateful_output.begin(), stateful_output.end(), n->output(0)->name())
            == stateful_output.end()) {
        CHECK(n->output_size() == 1);
        stateful_output.push_back(n->output(0)->name());
        if (std::find(out_edges.begin(), out_edges.end(), n->output(0)) !=
            out_edges.end()) {
          func_body += Ewise::EwiseGenBodyGetInput({n->output(0)}, false);
        }else {
          func_body += Ewise::EwiseGenBodyGetInput(n->output(0)->name(), 0.f);
        }
      }
      if (!n->IsStatefulOp())
        func_body +=  VarDeclStatementBuilder().SetNode(n).toCode();
      else
        func_body +=  AssignStatementBuilder().SetNode(n).toCode();
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
        .AttrList<string>("ZeroEnforced", stateful_output)
        .Device("GPU")
        .Finalize(&op_def);
      Node* new_node = new SingleNode(op_def, nodes.front()->scope());
      for (auto* e : out_edges) {
        new_node->AddOutput(e);
      }
      for (auto* e : in_edges) {
        new_node->AddInput(e);
      }
      parser_.AddFusedNode(new_node, i);
    }
    kernel_source_.push_back(source);
    in_edges.clear();
    out_edges.clear();
    nodes.clear();
  }

  for (auto &s : kernel_source_) 
    VLOG(V_DEBUG) << "KS: " << s;
  parser_.Finalize();
}

unordered_map<int, string> CodeGenerator::DataTypeToString =
    {{DT_FLOAT, "float"}, {DT_DOUBLE, "double"}, {DT_INT32, "int"}};

} //namespace RTC
} //namespace midend

