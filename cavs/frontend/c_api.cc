#include "c_api.h"

#include "cavs/midend/types.pb.h"
#include "cavs/midend/tensor_shape.pb.h"
#include "cavs/midend/devices.pb.h"
#include "cavs/midend/session.h"
#include "cavs/midend/devices.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/op.h"
#include "cavs/util/logging.h"
#include "cavs/frontend/dep_graph.h"
#include "cavs/frontend/node.h"

using midend::DataType;
using midend::SessionBase;
using midend::GetSession;
using midend::Tensor;
using midend::TensorShape;
using midend::TensorShapeDef;
using midend::GetAllocator;
using midend::ShapeInference;
using midend::DeviceTypeToString;
using frontend::DepGraph;
using frontend::Node;

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
      GetAllocator(DeviceTypeToString(midend::CPU)),
      midend::DataType((int)dtype),
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
  midend::OpDef op_def;
  op_def.ParseFromArray(def, def_length);
  Node* node = C_graph->graph->AddNode(op_def);
  TensorShapeDef shape_def;
  vector<const TensorShapeDef*> input_shapes;
  node->InputShapes(&input_shapes);
  ShapeInference(&shape_def, op_def, input_shapes);
  node->SetShape(shape_def);
  *dim_length = shape_def.dim_size();
  *dim = new int[*dim_length];
  for (int i = 0; i < shape_def.dim_size(); i++)
    (*dim)[i] = shape_def.dim(i);
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
