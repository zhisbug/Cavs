#ifndef CAVS_MIDEND_SESSION_H_
#define CAVS_MIDEND_SESSION_H_

#include "cavs/midend/tensor.h"
#include "cavs/frontend/dep_graph.h"

#include <unordered_map>

namespace midend {

class SessionBase {
 public:
  SessionBase(const ::frontend::DepGraph* graph) : graph_(graph) {}
  const Tensor* GetTensor(const string& name) const;
  void InsertTensor(const Tensor& t);
  virtual void Run(const vector<string>& output_names, 
                   vector<Tensor>* output_tensors,
                   const vector<string>& input_names,
                   const vector<Tensor>& input_tensors) {}
 protected:
  virtual void FeedInput(const vector<string>& input_names,
                   const vector<Tensor>& input_tensors) {}
  virtual void FetchOutput(const vector<string>& output_names,
                   vector<Tensor>* output_tensors) {}
  SessionBase() {}
  std::unordered_map<string, Tensor> tensor_map_;
  const ::frontend::DepGraph* graph_;
};

SessionBase* GetSession(const string& name, const ::frontend::DepGraph* graph);

#define REGISTER_SESSION_BUILDER(key, ...)                 \
    REGISTER_SESSION_BUILDER_UNIQ(__COUNTER__, key, __VA_ARGS__)
#define REGISTER_SESSION_BUILDER_UNIQ(ctr, key, ...)       \
    REGISTER_SESSION_BUILDER_CONCAT(ctr, key, __VA_ARGS__)
#define REGISTER_SESSION_BUILDER_CONCAT(ctr, key, ...)     \
    static session_factory::SessionRegister                \
        register_body_##ctr##_session(key,                 \
            [](const ::frontend::DepGraph* graph)          \
              -> SessionBase* {                            \
                return new __VA_ARGS__(graph);             \
              }) 

namespace session_factory {

class SessionRegister {
 public:
  typedef SessionBase* (*Factory)(const ::frontend::DepGraph* graph);
  SessionRegister(const string& name, Factory factory) {
    InitInternal(name, factory); 
  }
 private:
  void InitInternal(const string& name, Factory factory); 
};

} //namespace session_factory

} //namespace midend

#endif
