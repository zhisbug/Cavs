#include "c_api.h"

#include "cavs/proto/types.pb.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/proto/devices.pb.h"
#include "cavs/midend/session.h"
#include "cavs/midend/devices.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/dep_graph.h"
#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_impl.h"
#include "cavs/util/logging.h"

using midend::SessionBase;
using midend::GetSession;
using midend::Tensor;
using midend::TensorShape;
using midend::GetAllocator;
using midend::DeviceTypeToString;
using midend::DepGraph;
using midend::Node;
using backend::ShapeInference;

using std::string;
using std::vector;

namespace midend {

class TensorCApi {
 public:
  static void* raw_data(const Tensor& tensor) {
    return tensor.buf_->data();
  }
  static size_t size(const Tensor& tensor) {
    return tensor.buf_->size();
  }
};

}

struct C_Session {
  SessionBase* session;
};

struct C_DepGraph {
  DepGraph* graph;
};

struct C_Tensor {
  Tensor tensor;
};


C_Session* C_NewSessionWithDG(const char* name, size_t name_len, 
    C_DepGraph* C_graph) {
  string name_str(name, name_len);
  SessionBase* sess = GetSession(name_str, C_graph->graph);
  return new C_Session{sess};
}

C_Tensor* C_NewTensor(const char* name, size_t name_len, 
    const int* shape, int dims, C_Dtype dtype) {
  string name_str(name, name_len);
  TensorShape tshape;
  for (int i = 0; i < dims; i++)
    tshape.add_dim(shape[i]);
  Tensor t(name_str, 
      GetAllocator(DeviceTypeToString(CPU)), DataType((int)dtype),
      std::move(tshape));
  return new C_Tensor{t};
}

C_DepGraph* C_GetDefaultDG() {
  static C_DepGraph* dep_graph = new C_DepGraph{new DepGraph()};
  return dep_graph;
}

void C_AddNode(C_DepGraph* C_graph, 
    const void* def, size_t def_length,
    int** dim, size_t* dim_length) {
  OpDef op_def;
  op_def.ParseFromArray(def, def_length);
  Node* node = C_graph->graph->AddNode(op_def);
  vector<TensorShapeDef> input_shapes;
  node->InputShapes(&input_shapes);
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

void C_OptimizeWithLoss(C_DepGraph* c_graph, 
    const void* def, size_t def_length) {
  OpDef op_def;
  op_def.ParseFromArray(def, def_length);
  bool var_flag = false;
  for (auto& attr : op_def.attr()) {
    if (attr.name() == "vars") {
      CHECK(attr.value().list().s().size());
      var_flag = true;
    }
  }
  if (!var_flag) {
    vector<string> var_names;
    c_graph->graph->GroupAllVariables(&var_names);
    OpDef::AttrDef* var_attr = op_def.add_attr();
    var_attr->set_name("vars");
    OpDef::AttrType::ListValue* str_list
      = var_attr->mutable_value()->mutable_list();
    for (auto& var: var_names)
      str_list->add_s(var);
  }
  c_graph->graph->OptimizeWithLoss(op_def);
}

//void C_GetGrad(C_DepGraph* C_graph, 
      //const char* c_loss_name, int loss_name_len,
      //char** c_var_name, int var_name_len,
      //const char* c_proj_name, int proj_name_len,
      //int iters,
      //char *** c_grads, int* grads_num) {
  //string loss(c_loss_name, loss_name_len);
  //vector<string> var_names;
  //for (int i = 0; i < var_name_len; i++)
    //var_names.emplace_back(c_var_name[i]);
  //string proj(c_proj_name, proj_name_len);
  //vector<string> grads_vec;
  //C_graph->graph->BackPropagate(&grads_vec, loss);
  ////C_graph->graph->AddSolver("GradientDescent"+proj, var_names);
  //CHECK(grads_vec.size() == var_names.size());
  //*grads_num = grads_vec.size();
  //*c_grads = (char**)malloc(grads_vec.size()*sizeof(char*));
  //for (int i = 0; i < grads_vec.size(); i++) {
    //(*c_grads)[i] = (char*)malloc(grads_vec[i].length());
    //memcpy((*c_grads)[i], grads_vec[i].c_str(), grads_vec[i].length()+1);
  //}
//}
void C_DumpGraph(C_DepGraph* C_graph) {
  C_graph->graph->Dump();
}

void C_Run(C_Session* s, 
    const char** c_output_names, C_Tensor** c_output_tensors, int noutputs, 
    const char** c_input_names, C_Tensor* const* c_input_tensors, int ninputs) {
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
  }
}

void* C_TensorData(const C_Tensor* t) { 
  return midend::TensorCApi::raw_data(t->tensor); 
}

size_t C_TensorSize(const C_Tensor* t) { 
  return midend::TensorCApi::size(t->tensor); 
}

//C_Tensor* C_GetTensorFromSession(
      //C_Session* sess, const char* c_tensor_name, size_t len) {
  //string tensor_name(c_tensor_name, len);
  //const Tensor* t = sess->session->GetTensor(tensor_name);
  //CHECK_NOTNULL(t);
  //return new C_Tensor{const_cast<Tensor*>(t)};
//}
