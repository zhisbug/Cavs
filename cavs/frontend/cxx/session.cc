#include "cavs/frontend/cxx/session.h"

void Session::Run(const std::initializer_list<Sym>& outputs,
    const std::initializer_list<std::pair<Sym&, void*>>& feed) {
  vector<C_Tensor*> input_tensor;
  vector<const char*> input_name;
  for (auto& input : feed) {
    const Sym& sym = input.first;
    void* data = input.second;
    //for input, we assumpt they are all one-output operator
    if (feed_map_.count(sym.output(0)) == 0) {
      feed_map_[sym.output(0)] = 
        C_NewTensor(sym.output(0).c_str(), sym.output(0).length(), 
                    sym.shape().data(), sym.shape().size(),
                    sym.type());
    }
    C_Tensor* ft = feed_map_[sym.output(0)];
    memcpy(C_TensorData(ft), data, C_TensorSize(ft));
    input_tensor.push_back(ft);
    input_name.push_back(sym.output(0).data());
  }

  vector<C_Tensor*> output_tensor;
  vector<const char*> output_name;
  for (auto& fetch: outputs) {
    output_name.push_back(fetch.output(0).c_str());
  }
  output_tensor.resize(output_name.size(), NULL);
  C_Run(s_,
        output_name.data(),
        output_tensor.data(),
        output_name.size(),
        input_name.data(),
        input_tensor.data(),
        input_name.size());
  for (auto& fetch: outputs) {
    int i = 0;
    fetch.node_->raw_data = C_TensorData(output_tensor[i++]);
  }
  for (auto* t : output_tensor)
    free(t);
}