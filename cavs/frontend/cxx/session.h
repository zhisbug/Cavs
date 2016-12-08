#ifndef CAVS_FRONTEND_CXX_SESSION_H_
#define CAVS_FRONTEND_CXX_SESSION_H_

#include "cavs/frontend/c_api.h"
#include "cavs/frontend/cxx/sym.h"
#include "cavs/util/logging.h"
#include "cavs/midend/macros_gpu.h"

#include <string>
#include <initializer_list>
#include <unordered_map>
//#include <cuda_runtime.h>

class Session {
 public:
  Session(std::string name = "SimpleSession") {
    cavs::OpChainDef op_chain_def;
    Chain::Default()->Finalize(&op_chain_def);
    std::string serial_def;
    op_chain_def.SerializeToString(&serial_def);
    s_ = F_NewSession(name.c_str(), name.length(), 
                      serial_def.c_str(), serial_def.length());
  }

  //void Run(std::initializer_list<Sym> res) {
    //const char** output_names = (const char**)malloc(res.size()*sizeof(char*));
    //int i = 0;
    //for (auto sym : res) 
      //output_names[i++] = sym.output().c_str();
    //F_Tensor* output_tensors;
    //F_Run(s_, output_names, &output_tensors, res.size());
  //}

  typedef std::initializer_list<std::pair<Sym&, void*>> FEED;
  void Run(Sym& res, FEED feed) {
    vector<F_Tensor*> input_tensor;
    vector<const char*> input_name;
    for (auto& input : feed) {
      Sym& sym = input.first;
      void* data = input.second;
      if (feed_map_.count(sym.output()) == 0) {
        feed_map_[sym.output()] = 
          F_NewTensor(sym.output().c_str(), sym.output().length(), 
                      sym.shape().data(), sym.shape().size(),
                      sym.type());
        //F_Tensor* t = feed_map_[input_name];
        ////currently, we don't support inter device mechnism, 
        ////we hack it here.
        //cudaMemcpy(F_TensorData(t), data, F_TensorSize(t), cudaMemcpyHostToDevice);
      }
      F_Tensor* ft = feed_map_[sym.output()];
      memcpy(F_TensorData(ft), data, F_TensorSize(ft));
      input_tensor.push_back(ft);
      input_name.push_back(input.first.output().data());
    }
    vector<F_Tensor*> output_tensor(1);
    vector<const char*> output_name(1);
    output_name[0] = res.output().c_str();
    if (fetch_map_.count(res.output()) == 0) {
      fetch_map_[res.output()] = 
        F_NewTensor(res.output().c_str(), res.output().length(), 
                    res.shape().data(), res.shape().size(),
                    res.type());
    }
    output_tensor[0] = fetch_map_[res.output()];
    F_Run(s_, output_name.data(), output_tensor.data(), 1,
          input_name.data(), input_tensor.data(), input_name.size());
    //checkCudaError(cudaMemcpy(res.body_->raw_data, F_TensorData(output_tensor[0]), 
               //F_TensorSize(output_tensor[0]), cudaMemcpyDeviceToHost));
    res.body_->raw_data = F_TensorData(output_tensor[0]);
  }

 private:
  F_Session* s_;
  std::unordered_map<string, F_Tensor*> feed_map_;
  std::unordered_map<string, F_Tensor*> fetch_map_;
};

#endif
