#ifndef CAVS_MIDEND_SESSION_H_
#define CAVS_MIDEND_SESSION_H_

#include "cavs/midend/session_base.h"
#include "cavs/midend/statement.h"

namespace midend {

class SimpleSession : public SessionBase {
 public:
  SimpleSession(const DepGraph* graph);
  void Run(const std::vector<std::string>& output_names, 
           std::vector<Tensor>* output_tensors,
           const std::vector<std::string>& input_names,
           const std::vector<Tensor>& input_tensors) override;
 protected:
  virtual void Compile(const std::vector<std::string>& output_names, 
                       const std::vector<std::string>& input_names);
  void FeedInput(const std::vector<std::string>& input_names,
                 const std::vector<Tensor>& input_tensors);
  void FetchOutput(const std::vector<std::string>& output_names,
                   std::vector<Tensor>* output_tensors);
  //std::vector<std::pair<OpImpl*, OpContext*>> executors_;
  std::vector<Statement*> executors_;
  bool compiled_;
  int round_;//current batch id;
};

class MPISession: public SimpleSession {
 public:
  MPISession(const DepGraph* graph);
  void Run(const std::vector<std::string>& output_names, 
           std::vector<Tensor>* output_tensors,
           const std::vector<std::string>& input_names,
           const std::vector<Tensor>& input_tensors) override;
 private:
  void Compile(const std::vector<std::string>& output_names, 
               const std::vector<std::string>& input_names) override;
  std::vector<Statement*> executors_;
  bool compiled_;
  int round_;
};

} //namespace midend

#endif
