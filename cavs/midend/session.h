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
  int SessionType() override { return SIMPLE; }
 protected:
  virtual void Compile(const std::vector<std::string>& output_names);
  void FeedInput(const std::vector<std::string>& input_names,
                 const std::vector<Tensor>& input_tensors);
  void FetchOutput(const std::vector<std::string>& output_names,
                   std::vector<Tensor>* output_tensors);
  //std::vector<Statement*> executors_;
  std::unordered_map<std::string, std::vector<Statement*>> executors_;
  int round_;//current batch id;
};

class MPISession: public SimpleSession {
 public:
  MPISession(const DepGraph* graph);
  ~MPISession();
  void Run(const std::vector<std::string>& output_names, 
           std::vector<Tensor>* output_tensors,
           const std::vector<std::string>& input_names,
           const std::vector<Tensor>& input_tensors) override;
  int SessionType() override { return MPI; }
 private:
  void Compile(const std::vector<std::string>& output_names) override;
};

} //namespace midend

#endif
