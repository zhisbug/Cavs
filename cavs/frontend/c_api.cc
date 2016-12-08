#include "c_api.h"

#include "cavs/midend/op_chain_def.pb.h"
#include "cavs/midend/types.pb.h"
#include "cavs/midend/devices.pb.h"
#include "cavs/midend/session.h"
#include "cavs/midend/devices.h"
#include "cavs/midend/tensor.h"
#include "cavs/util/logging.h"

using cavs::DataType;
using cavs::OpChainDef;
using cavs::SessionBase;
using cavs::GetSession;
using cavs::Tensor;
using cavs::TensorShape;
using cavs::GetAllocator;
using cavs::DeviceTypeToString;

namespace cavs {

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

struct F_Session {
  SessionBase* session;
};
struct F_Tensor {
  Tensor* tensor;
};

//F_Session* F_NewSession(const char* name, size_t len) {
F_Session* F_NewSession(const char* name, size_t name_len, 
    const void* proto, size_t proto_len) {
  string name_str(name, name_len);
  OpChainDef def;
  def.ParseFromArray(proto, proto_len);
  SessionBase* sess = GetSession(name_str, def);
  return new F_Session{sess};
}

F_Tensor* F_NewTensor(const char* name, size_t name_len, 
    const int* shape, int dims, F_Dtype dtype) {
  string name_str(name, name_len);
  TensorShape tshape;
  for (int i = 0; i < dims; i++)
    tshape.add_dim(shape[i]);
  Tensor *t = new Tensor(name_str, 
      GetAllocator(DeviceTypeToString(cavs::CPU)),
      cavs::DataType((int)dtype),
      std::move(tshape));
  return new F_Tensor{t};
}

//void F_SetOpChainOp(F_Session* s, 
                    //const void* proto, size_t len) {
  //OpChainDef def;
  //def.ParseFromArray(proto, len);
  //s->session->SetOpChainDef(def);
//}

void F_Run(F_Session* s, 
    const char** c_output_names, F_Tensor** c_output_tensors, int noutputs, 
    const char** c_input_names, F_Tensor** c_input_tensors, int ninputs) {
  vector<string> output_names(noutputs);
  vector<const Tensor*> output_tensors(noutputs);
  for (int i = 0; i < noutputs; i++)
    output_names[i] = c_output_names[i];
  vector<string> input_names(ninputs);
  vector<const Tensor*> input_tensors(ninputs);
  for (int i = 0; i < ninputs; i++)
    input_names[i] = c_input_names[i];

  s->session->Run(output_names, &output_tensors, input_names, input_tensors);
  for (int i = 0; i < noutputs; i++) {
    c_output_tensors[i]->tensor = 
        const_cast<Tensor*>(output_tensors[i]);
  }
}

void* F_TensorData(const F_Tensor* t) { 
  return const_cast<void*>(cavs::TensorCApi::raw_data(*(t->tensor))); 
}

size_t F_TensorSize(const F_Tensor* t) { 
  return (cavs::TensorCApi::size(*(t->tensor))); 
}

F_Tensor* F_GetTensorFromSession(
      F_Session* sess, const char* c_tensor_name, size_t len) {
  string tensor_name(c_tensor_name, len);
  const Tensor* t = sess->session->GetTensor(tensor_name);
  CHECK_NOTNULL(t);
  return new F_Tensor{const_cast<Tensor*>(t)};
}

//memcpy(TensorCApi::raw_data(*tensor), data, len);
//return new 
