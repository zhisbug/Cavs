#include "c_api.h"

#include "cavs/midend/session.h"
#include "cavs/midend/op_chain_def.pb.h"

using cavs::SessionBase;
using cavs::OpChainDef;
using cavs::GetSession;
using cavs::Tensor;

struct F_Session {
  SessionBase* session;
};
struct F_Tensor {
  Tensor* tensor;
};

F_Session* F_NewSession(const char* name, size_t len) {
  string name_str(name, len);
  SessionBase* sess = GetSession(name_str);
  return new F_Session{sess};
}

void F_SetOpChainOp(F_Session* s, 
                    const void* proto, size_t len) {
  OpChainDef def;
  def.ParseFromArray(proto, len);
  s->session->SetOpChainDef(def);
}

void F_Run(F_Session* s, const char** c_output_names,
           F_Tensor** c_output_tensors, int noutputs) {
  vector<string> output_names(noutputs);
  for (int i = 0; i < noutputs; i++)
    output_names[i] = c_output_names[i];
  vector<const Tensor*> output_tensors(noutputs);
  s->session->Run(output_names, &output_tensors);
  for (int i = 0; i < noutputs; i++) {
    c_output_tensors[i]->tensor = 
        const_cast<Tensor*>(output_tensors[i]);
  }
}

void* F_TensorData(const F_Tensor* t) { 
  return const_cast<void*>(t->tensor->raw_data()); 
}
