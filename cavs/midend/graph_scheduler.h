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
    batch_size_(0), max_seq_length_(0), total_length_(0) {}
  virtual void Initialize() = 0;
  virtual int  JobIdToInternalTensorId(int gid) const = 0;
  virtual int  InternalTensorIdToJobId(int tid) const = 0;
  virtual bool Terminate() const = 0;
  inline virtual void ActivateNext();
  inline virtual int GetCurrentRoundOffset() const;

  int LoadGraph(const Tensor& parent_ids);
  int ReverseGraph();
  inline const std::vector<int>& GetJobId() const;
  inline int batch_size() const { return batch_size_; }
  inline bool HasChild(int job_id) const;
  inline const std::vector<int>& child_id(int job_id) const;

  inline void SetMessagePasser(const Tensor& t) {
    CHECK(!t.IsFullShape());
    message_passer_ = t; 
  }
  inline const Tensor& GetMessagePasser(int id) {
    message_passer_.SetOffsetWithId(id);
    return message_passer_; 
  }
  inline void SetFuncArg(const Tensor t) { 
    //we loose this constraint because of the label reshape
    //CHECK(t.IsFullShape());
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
    //return sample_id*max_seq_length_+local_id; 
    return sample_offset_in_gid_[sample_id]+local_id;
  }
  std::vector<int>  sample_offset_in_gid_;
  std::vector<int>  ready_to_execute_ids_;
  std::vector<int> activated_times_;
  int batch_size_;
  int max_seq_length_;
  int total_length_;
  Tensor message_passer_;
  Tensor func_arg_;
  Tensor func_ret_;
  std::vector<std::vector<int>> *parents_;
  std::vector<std::vector<int>> *children_;
  std::vector<int> round2offset_;
  struct RoundCounter {
   public:
    RoundCounter() : round_(-1), isforward_(true) {}
    void Reset() { round_ = -1; isforward_ = true; }
    void SetBackward() { isforward_ = false; }
    bool IsForward() const { return isforward_; }
    int operator()() const { return round_; }
    RoundCounter& operator ++() {
      if (isforward_) round_++; 
      else round_--; 
      return *this;
    }

   private: 
    int round_;
    bool isforward_;
  };
  RoundCounter rc_;

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
  inline int JobIdToInternalTensorId(int gid) const override { return gid; }
  inline int InternalTensorIdToJobId(int tid) const override { return tid; }
  inline int GetCurrentRoundOffset() const override { return GetJobId()[0]; }

 private:
  //inline int toSampleId(int gid) const {
    //return gid/max_seq_length_;
  //}
  int sample_id_;
  void InitializeSample(int id);
  std::list<int> pending_list_;
};

class BatchGraphScheduler : public GraphSchedulerBase {
 public:
  BatchGraphScheduler() :
    GraphSchedulerBase(), job2intensor_(0), intensor2job_(0), execution_tracer_(0) {}
  void Initialize() override;
  void ActivateNext() override;
  inline bool Terminate() const override { return ready_to_execute_ids_.empty(); }
  inline int JobIdToInternalTensorId(int gid) const override { return job2intensor_[gid]; }
  inline int InternalTensorIdToJobId(int tid) const override { return intensor2job_[tid]; }

 private:
  std::vector<int> job2intensor_;
  std::vector<int> intensor2job_;
  std::vector<std::vector<int>> execution_tracer_;
};

inline void GraphSchedulerBase::ActivateNext() {
  ++rc_;
  if (round2offset_.size() <= rc_())
    round2offset_.push_back(round2offset_.back() + GetJobId().size());
}

inline bool GraphSchedulerBase::HasChild(int job_id) const {
  CHECK(job_id < (*children_).size());
  return !(*children_)[job_id].empty();
}

inline const std::vector<int>& GraphSchedulerBase::child_id(int job_id) const {
  CHECK(HasChild(job_id));
  return (*children_)[job_id];
}

inline const std::vector<int>& GraphSchedulerBase::GetJobId() const {
  CHECK(!Terminate());
  return ready_to_execute_ids_;
}

inline int GraphSchedulerBase::GetCurrentRoundOffset() const {
  return round2offset_[rc_()]; 
}

} //namespace midend

#endif

