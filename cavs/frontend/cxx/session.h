#ifndef CAVS_FRONTEND_CXX_SESSION_H_
#define CAVS_FRONTEND_CXX_SESSION_H_

#include "cavs/frontend/c_api.h"
#include "cavs/frontend/cxx/sym.h"

#include <string>
#include <initializer_list>

class Session {
 public:
  Session(std::string name = "SimpleSession") {
    s_ = F_NewSession(name.c_str(), name.length());
    cavs::OpChainDef op_chain_def;
    Chain::Default()->Finalize(&op_chain_def);
    std::string serial_def;
    op_chain_def.SerializeToString(&serial_def);
    F_SetOpChainOp(s_, serial_def.c_str(), serial_def.length());
  }

  //void Run(std::initializer_list<Sym> res) {
    //const char** output_names = (const char**)malloc(res.size()*sizeof(char*));
    //int i = 0;
    //for (auto sym : res) 
      //output_names[i++] = sym.output().c_str();
    //F_Tensor* output_tensors;
    //F_Run(s_, output_names, &output_tensors, res.size());
  //}

  Sym& Run(Sym& res) {
    F_Tensor* output_tensor;
    const char* output_name = res.body_->output_.c_str();
    F_Run(s_, &output_name, &output_tensor, 1);
    res.body_->raw_data = F_TensorData(output_tensor);
    return res;
  }

 private:
  F_Session* s_;
};

#endif
