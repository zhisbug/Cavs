#ifndef CAVS_MIDEND_SESSION_BASE_H_
#define CAVS_MIDEND_SESSION_BASE_H_

#include "cavs/midend/tensor.h"
#include "cavs/midend/node.h"

#include <unordered_map>

namespace midend {

class OpContext;
class Node;
class SessionBase {
 public:
  const Tensor* GetTensor(const std::string& name, bool recursive = false) const;
  void InsertTensor(const Tensor& t);
  virtual void Run(const std::vector<std::string>& output_names, 
                   std::vector<Tensor>* output_tensors,
                   const std::vector<std::string>& input_names,
                   const std::vector<Tensor>& input_tensors) {}
  virtual OpContext* GetContext(const Node* node);
  virtual int SessionType() { return BASE; }
  enum { BASE=1, SIMPLE=2, MPI=3, PARTIALEXECUTION=4 };
  std::string DebugInfo();
 protected:
  std::unordered_map<std::string, Tensor> tensor_map_;
};

SessionBase* GetSession(const std::string& name);

#define REGISTER_SESSION_BUILDER(key, ...)                 \
    REGISTER_SESSION_BUILDER_UNIQ(__COUNTER__, key, __VA_ARGS__)
#define REGISTER_SESSION_BUILDER_UNIQ(ctr, key, ...)       \
    REGISTER_SESSION_BUILDER_CONCAT(ctr, key, __VA_ARGS__)
#define REGISTER_SESSION_BUILDER_CONCAT(ctr, key, ...)     \
    static session_factory::SessionRegister                \
        register_body_##ctr##_session(key,                 \
            []() -> SessionBase* {                            \
                return new __VA_ARGS__();             \
              }) 

namespace session_factory {

class SessionRegister {
 public:
  typedef SessionBase* (*Factory)();
  SessionRegister(const std::string& name, Factory factory) {
    InitInternal(name, factory); 
  }
 private:
  void InitInternal(const std::string& name, Factory factory); 
};

} //namespace session_factory

} //namespace midend

#endif
