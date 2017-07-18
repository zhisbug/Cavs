#ifndef CAVS_MIDEND_GRAPH_SCHEDULER_H_
#define CAVS_MIDEND_GRAPH_SCHEDULER_H_

#include "cavs/midend/tensor.h"
#include "cavs/util/logging.h"

#include <vector>
#include <list>

namespace midend {

class GraphScheduler {
 public:
  GraphScheduler() : parent_ids_(0), batch_(0), sample_id_(0), isForward_(true) {}
  void ActivateNext();
  int LoadGraph(const Tensor& parent_ids);
  void ReverseGraph();
  int GetJobId() const;
  inline bool empty() const { return pending_workset_.empty(); }

  inline void SetMessagePasser(const Tensor& t) { message_passer_ = t; }
  inline const Tensor& GetMessagePasser(int id) {
    message_passer_.SetOffsetWithId(id);
    return message_passer_; 
  }
  inline void SetFuncArg(const Tensor t) { 
    CHECK(t.IsFullShape());
    func_arg_ = t;
  }
  inline const Tensor& GetFuncArg() {
    return func_arg_; 
  }
  inline void SetFuncRet(const Tensor t) { 
    CHECK(!t.IsFullShape());
    func_ret_ = t;
  }
  inline const Tensor& GetFuncRet() {
    func_ret_.SetOffsetWithId(0);
    return func_ret_; 
  }

  bool HasChild() const;
  std::vector<int> child_id() const;

  inline int batch() const { return batch_; }
  void TrigerBatchId(int sample_id);

 private:
  int ToGlobalId(int local_id) const;
  std::list<int> pending_workset_;
  std::vector<bool> activated_workset_;
  std::vector<std::vector<int>> parent_ids_;
  std::vector<std::vector<std::vector<int>>> child_ids_;
  int sample_id_;
  int batch_;
  bool isForward_;
  Tensor message_passer_;
  Tensor func_arg_;
  Tensor func_ret_;
};

class BatchedGraphScheduler : public GraphScheduler {

};

} //namespace midend

#endif

