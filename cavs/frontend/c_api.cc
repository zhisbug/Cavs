#include "c_api.h"

#include "cavs/proto/types.pb.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/proto/devices.pb.h"
#include "cavs/proto/func_def.pb.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/midend/session_base.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/scope.h"
#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_impl.h"
#include "cavs/util/logging.h"

using midend::SessionBase;
using midend::GetSession;
using midend::Tensor;
using midend::TensorShape;
using midend::GetAllocator;
using midend::Scope;
using midend::main_scope;
//using midend::global_scope;
using midend::SingleNode;
using backend::ShapeInference;

using std::string;
using std::vector;

namespace midend {

class TensorCApi {
 public:
  static void* raw_data(const Tensor& tensor) {
    CHECK(tensor.buf_) << tensor.debug_info();
    return tensor.buf_->data();
  }
  static size_t size(const Tensor& tensor) {
    CHECK(tensor.buf_) << tensor.debug_info();
    return tensor.buf_->size();
  }
  static bool IsVirtual(const Tensor& tensor) {
    return tensor.empty();  
  }
};

}

struct C_Session {
  SessionBase* session;
};

struct C_Scope {
  Scope* scope;
};

struct C_Tensor {
  Tensor tensor;
};


C_Session* C_NewSession(const char* name, size_t name_len, int opt) {
  string name_str(name, name_len);
  //SessionBase* sess = GetSession(name_str, C_graph->graph);
  SessionBase* sess = GetSession(name_str, opt);
  return new C_Session{sess};
}

C_Tensor* C_NewTensor(const char* name, size_t name_len, 
    const int* shape, int dims, C_Dtype dtype) {
  string name_str(name, name_len);
  TensorShape tshape;
  for (int i = 0; i < dims; i++)
    tshape.AddDim(shape[i]);
  Tensor t(name_str, 
      GetAllocator(DeviceTypeToString(CPU)), DataType((int)dtype),
      std::move(tshape));
  return new C_Tensor{t};
}

C_Scope* C_GetMainScope() {
  static C_Scope* scope = new C_Scope{ main_scope() };
  return scope;
}

//C_Scope* C_GetGlobalScope() {
  //static C_Scope* scope = new C_Scope{ global_scope() };
  //return scope;
//}

void C_AddOp(const void* def, size_t def_length,
    int** dim, size_t* dim_length) {
  OpDef op_def;
  op_def.ParseFromArray(def, def_length);
  SingleNode* node = C_GetMainScope()->scope->AddOp(op_def);
  const vector<TensorShapeDef>& input_shapes =
    node->input_shapes();
  const vector<TensorShapeDef>& shape_def =
    ShapeInference(op_def, input_shapes);
  node->SetShape(shape_def);
  //for user interface, the output of each operator can only be 1
  CHECK(shape_def.size() == 1);
  *dim_length = shape_def[0].dim_size();
  *dim = new int[*dim_length];
  for (int i = 0; i < shape_def[0].dim_size(); i++)
    (*dim)[i] = shape_def[0].dim(i);
}

void C_AddFunction(const void* def, size_t def_length,
    int** dim, size_t* dim_length) {
  FunctionDef func_def;
  func_def.ParseFromArray(def, def_length);
  TensorShapeDef shape = C_GetMainScope()->scope->AddFunction(func_def);
  *dim_length = shape.dim_size();
  *dim = new int[*dim_length];
  for (int i = 0; i < shape.dim_size(); i++)
    (*dim)[i] = shape.dim(i);
}

void C_AddOptimizerOp(const void* def, size_t def_length) {
  OpDef op_def;
  op_def.ParseFromArray(def, def_length);
  bool hasVars = false;
  for (auto& attr : *op_def.mutable_attr()) {
    if (attr.name() == "Vars") {
      CHECK(!hasVars);
      if (attr.value().list().s().size() == 0) {
        vector<string> var_names;
        C_GetMainScope()->scope->GroupAllVariables(&var_names);
        OpDef::AttrType::ListValue* str_list
          = attr.mutable_value()->mutable_list();
        for (auto& var: var_names)
          str_list->add_s(var);
      }
      hasVars = true;
    }
  }
  CHECK(hasVars);
  //c_graph->graph->OptimizeWithLoss(op_def);
  VLOG(V_DEBUG) << "Adding Optimizer\n" << op_def.DebugString();
  C_GetMainScope()->scope->AddOptimizerOp(op_def);
}

void C_AddControlDependency(const void* def, size_t def_length) {
  OpDef op_def;
  op_def.ParseFromArray(def, def_length);
  C_GetMainScope()->scope->AddControlDependency(op_def);
}

void C_Run(C_Session* s, 
    const char** c_output_names, C_Tensor** c_output_tensors, int noutputs, 
    const char** c_input_names, C_Tensor* const* c_input_tensors, int ninputs) {
  VLOG(V_DEBUG) << "Session Run Begins...\n";
  vector<string> output_names(noutputs);
  vector<Tensor> output_tensors(noutputs);
  for (int i = 0; i < noutputs; i++) {
    output_names[i] = c_output_names[i];
  }
  vector<string> input_names(ninputs);
  vector<Tensor> input_tensors(ninputs);
  for (int i = 0; i < ninputs; i++) {
    input_names[i] = c_input_names[i];
    input_tensors[i] = c_input_tensors[i]->tensor;
  }

  s->session->Run(output_names, &output_tensors, 
                  input_names, input_tensors);
  for (int i = 0; i < noutputs; i++) {
    c_output_tensors[i] = new C_Tensor{output_tensors[i]};
    //if (C_TensorData(c_output_tensors[i]))
      //LOG(INFO) << i << "\t" << *(float*)C_TensorData(c_output_tensors[i]);
  }
}

void* C_TensorData(const C_Tensor* t) { 
  CHECK(t);
  if (midend::TensorCApi::IsVirtual(t->tensor))
    return NULL;
  else 
    return midend::TensorCApi::raw_data(t->tensor); 
}

size_t C_TensorSize(const C_Tensor* t) { 
  if (midend::TensorCApi::IsVirtual(t->tensor))
    return 0;
  else
    return midend::TensorCApi::size(t->tensor); 
}

