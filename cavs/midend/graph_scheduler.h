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
    batch_size_(0), max_seq_length_(0), total_length_(0), gpu_idx_buf_(NULL) {
      tids_for_gather_init_.resize(2);
      tids_for_gather_.resize(2);
      tids_for_scatter_.resize(2);
  }
  virtual void Initialize() = 0;
  virtual bool Terminate() const = 0;
  virtual void ActivateNext() = 0;
  virtual int GetCurrentRoundOffset() const = 0;

  int LoadGraph(const Tensor& parent_ids);
  int ReverseGraph();
  inline int batch_size() const { return batch_size_; }
  inline int total_length() const { return total_length_; }
  inline int* gpu_idx_buf() const { return gpu_idx_buf_; }
  inline bool HasChild(int job_id) const {
    CHECK(job_id < (*children_).size());
    return !(*children_)[job_id].empty();
  }
  inline const std::vector<int>& GetJobId() const {
    CHECK(!Terminate());
    return ready_to_execute_ids_;
  }
  inline const std::vector<int>& CurrentRoundTensorIdsForGatherInitialization() const {
    if (rc_.IsForward())
      return tids_for_gather_init_[0];
    else
      return tids_for_gather_init_[1];
  }
  inline const std::vector<int>& CurrentRoundTensorIdsForGather(int child_offset) const {
    return tids_for_gather_[child_offset];
  }
  inline const std::vector<int>& CurrentRoundTensorIdsForScatter(int child_offset) const {
    return tids_for_scatter_[child_offset]; 
  }
  inline const std::vector<int>& TensorIdsToJobIds() const {
    return tids_to_jobids_; 
  }

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
    return sample_offset_in_gid_[sample_id]+local_id;
  }
  std::vector<int>  sample_offset_in_gid_;
  std::vector<int>  ready_to_execute_ids_;
  std::vector<int>  activated_times_;
  std::vector<std::vector<int>> tids_for_gather_init_;
  std::vector<std::vector<int>> tids_for_gather_;
  std::vector<std::vector<int>> tids_for_scatter_;
  std::vector<int> jobids_to_tids_;
  std::vector<int> tids_to_jobids_;
  std::vector<int> round2offset_;

  Tensor message_passer_;
  Tensor func_arg_;
  Tensor func_ret_;
  std::vector<std::vector<int>> *parents_;
  std::vector<std::vector<int>> *children_;
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
  int max_seq_length_;
  int batch_size_;
  int total_length_;
  std::vector<std::vector<int>> __forward_parents_ids_;
  std::vector<std::vector<int>> __forward_children_ids_;
  int* gpu_idx_buf_;
};

class SerialGraphScheduler : public GraphSchedulerBase {
 public:
  SerialGraphScheduler() : GraphSchedulerBase() {
    ready_to_execute_ids_.resize(1);
  }
  void Initialize() override;
  void ActivateNext() override;
  inline bool Terminate() const override { return pending_list_.empty(); }
  inline int GetCurrentRoundOffset() const override { return GetJobId()[0]; }

 private:
  int sample_id_;
  void InitializeSample(int id);
  std::list<int> pending_list_;
};

class BatchGraphScheduler : public GraphSchedulerBase {
 public:
  BatchGraphScheduler() :
    GraphSchedulerBase(), execution_tracer_(0) {}
  void Initialize() override;
  void ActivateNext() override;
  inline bool Terminate() const override { return ready_to_execute_ids_.empty(); }
  inline int GetCurrentRoundOffset() const override { return round2offset_[rc_()]; }

 private:
  std::vector<std::vector<int>> execution_tracer_;
  std::vector<std::vector<std::vector<int>>> gather_tracer_;
  std::vector<std::vector<std::vector<int>>> scatter_tracer_;
};


} //namespace midend

#endif

