#ifndef CAVS_MIDEND_GRAPH_SCHEDULER_H_
#define CAVS_MIDEND_GRAPH_SCHEDULER_H_

#include "cavs/midend/tensor.h"
#include "cavs/util/logging.h"

#include <vector>
#include <list>

namespace midend {

class GraphSchedulerBase {
 public:
  GraphSchedulerBase() :
    batch_size_(0), max_seq_length_(0) {}
  virtual void Initialize() = 0;
  virtual void ActivateNext() = 0;
  virtual int  JobIdToTensorId(int gid) const = 0;
  virtual int  GetCurrentRoundOffset() const = 0;
  virtual bool Terminate() const = 0;

  int LoadGraph(const Tensor& parent_ids);
  void ReverseGraph();
  inline const std::vector<int>& GetJobId() const;
  inline int batch_size() const { return batch_size_; }
  inline bool HasChild(int job_id) const;
  inline const std::vector<int>& child_id(int job_id) const;

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

 protected:
  inline int toGlobalId(int sample_id, int local_id) const {
    return sample_id*max_seq_length_+local_id; 
  }
  inline int toSampleId(int gid) const {
    return gid/max_seq_length_;
  }
  std::vector<int>  ready_to_execute_ids_;
  std::vector<bool> activated_ids_;
  int batch_size_;
  int max_seq_length_;
  Tensor message_passer_;
  Tensor func_arg_;
  Tensor func_ret_;
  std::vector<std::vector<int>> *parents_;
  std::vector<std::vector<int>> *children_;

 private:
  std::vector<std::vector<int>> __forward_parents_ids_;
  std::vector<std::vector<int>> __forward_children_ids_;
};

class SerialGraphScheduler : public GraphSchedulerBase {
 public:
  SerialGraphScheduler() : GraphSchedulerBase() {
    ready_to_execute_ids_.resize(1);
  }
  void Initialize() override;
  void ActivateNext() override;
  inline bool Terminate() const override { return pending_list_.empty(); }
  inline int JobIdToTensorId(int gid) const override { return gid; }
  inline int GetCurrentRoundOffset() const override { return GetJobId()[0]; }

 private:
  void InitializeSample(int id);
  std::list<int> pending_list_;
};

class BatchGraphScheduler : public GraphSchedulerBase {
 public:
  BatchGraphScheduler() :
    GraphSchedulerBase(), executed_jobs_(0) {}
  void Initialize() override;
  void ActivateNext() override;
  inline bool Terminate() const override { return ready_to_execute_ids_.empty(); }
  inline int JobIdToTensorId(int gid) const override { return job2tensor_[gid]; }
  inline int GetCurrentRoundOffset() const override { return executed_jobs_; }

 private:
  int executed_jobs_;
  std::vector<int> job2tensor_;
};

inline bool GraphSchedulerBase::HasChild(int job_id) const {
  //CHECK(!Terminate());
  //if (isForward_) {
    //CHECK(job_id < child_ids_.size());
    //return !child_ids_[job_id].empty();
  //}else{
    //CHECK(job_id < parent_ids_.size());
    //return (!parent_ids_[job_id].emtpy());
  //}
  CHECK(job_id < (*children_).size());
  return !(*children_)[job_id].empty();
}

inline const std::vector<int>& GraphSchedulerBase::child_id(int job_id) const {
  //CHECK(!Terminate());
  CHECK(HasChild(job_id));
  //if (isForward_) {
    //return child_ids_[job_id];
  //}else {
    //return parent_ids_[job_id];
  //}
  return (*children_)[job_id];
}

const std::vector<int>& GraphSchedulerBase::GetJobId() const {
  CHECK(!Terminate());
  return ready_to_execute_ids_;
}

} //namespace midend

#endif

