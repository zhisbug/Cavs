#ifndef CAVS_FRONTEND_CXX_SESSION_H_
#define CAVS_FRONTEND_CXX_SESSION_H_

#include "cavs/frontend/c_api.h"
#include "cavs/frontend/cxx/sym.h"
#include "cavs/util/logging.h"
#include "cavs/util/macros_gpu.h"

#include <string>
#include <initializer_list>
#include <unordered_map>

class Session {
 public:
  Session(std::string name = "SimpleSession") {
    s_ = C_NewSessionWithDG(
        name.c_str(), name.length(), C_GetDefaultDG());
  }

  //void Run(std::initializer_list<Sym> res) {
    //const char** output_names = (const char**)malloc(res.size()*sizeof(char*));
    //int i = 0;
    //for (auto sym : res) 
      //output_names[i++] = sym.output().c_str();
    //C_Tensor* output_tensors;
    //C_Run(s_, output_names, &output_tensors, res.size());
  //}

  void Run(Sym& out, std::initializer_list<std::pair<Sym&, void*>> feed) {
    vector<C_Tensor*> input_tensor;
    vector<const char*> input_name;
    for (auto& input : feed) {
      const Sym& sym = input.first;
      void* data = input.second;
      if (feed_map_.count(sym.output()) == 0) {
        feed_map_[sym.output()] = 
          C_NewTensor(sym.output().c_str(), sym.output().length(), 
                      sym.shape().data(), sym.shape().size(),
                      sym.type());
      }
      C_Tensor* ft = feed_map_[sym.output()];
      memcpy(C_TensorData(ft), data, C_TensorSize(ft));
      input_tensor.push_back(ft);
      input_name.push_back(sym.output().data());
    }
    vector<C_Tensor*> output_tensor(1);
    vector<const char*> output_name(1);
    output_name[0] = out.output().c_str();
    C_Run(s_, output_name.data(), output_tensor.data(), 1,
          input_name.data(), input_tensor.data(), input_name.size());
    out.body_->raw_data = C_TensorData(output_tensor[0]);
    for (auto* t : output_tensor)
      free(t);
  }

 private:
  C_Session* s_;
  std::unordered_map<string, C_Tensor*> feed_map_;
};

#endif
