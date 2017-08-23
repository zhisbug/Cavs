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
      tids_for_gather_.resize(2);
      tids_for_scatter_.resize(2);
  }
  virtual void Initialize() = 0;
  virtual bool Terminate() const = 0;
  virtual void ActivateNext() = 0;
  virtual int GetCurrentRoundOffset() const = 0;
  //virtual std::vector<int> JobIdToInternalTensorIds(const std::vector<int>& gids) const = 0;
  //virtual std::vector<int> InternalTensorIdToJobIds(const std::vector<int>& tids) const = 0;

  int LoadGraph(const Tensor& parent_ids);
  int ReverseGraph();
  inline int batch_size() const { return batch_size_; }
  inline int* gpu_idx_buf() const { return gpu_idx_buf_; }
  inline bool HasChild(int job_id) const {
    CHECK(job_id < (*children_).size());
    return !(*children_)[job_id].empty();
  }
  inline const std::vector<int>& GetJobId() const {
    CHECK(!Terminate());
    return ready_to_execute_ids_;
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
    //return sample_id*max_seq_length_+local_id; 
    return sample_offset_in_gid_[sample_id]+local_id;
  }
  std::vector<int>  sample_offset_in_gid_;
  std::vector<int>  ready_to_execute_ids_;
  std::vector<int>  activated_times_;
  std::vector<std::vector<int>> tids_for_gather_;
  std::vector<std::vector<int>> tids_for_scatter_;
  std::vector<int> jobids_to_tids_;
  std::vector<int> tids_to_jobids_;
  std::vector<int> round2offset_;
  //std::vector<int> intensor2job_;

  int batch_size_;
  int max_seq_length_;
  int total_length_;
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
  //inline int JobIdToInternalTensorId(int gid) const override { return gid; }
  //inline int InternalTensorIdToJobId(int tid) const override { return tid; }
  //std::vector<int> JobIdToInternalTensorIds(const std::vector<int>& gids) const override { return gids; }
  //std::vector<int> InternalTensorIdToJobIds(const std::vector<int>& tids) const override { return tids; }

 private:
  int sample_id_;
  void InitializeSample(int id);
  std::list<int> pending_list_;
};

class BatchGraphScheduler : public GraphSchedulerBase {
 public:
  BatchGraphScheduler() :
    GraphSchedulerBase(), /*job2intensor_(0), intensor2job_(0),*/ execution_tracer_(0) {}
  void Initialize() override;
  void ActivateNext() override;
  inline bool Terminate() const override { return ready_to_execute_ids_.empty(); }
  inline int GetCurrentRoundOffset() const override { return round2offset_[rc_()]; }
  //inline int JobIdToInternalTensorId(int gid) const override { return job2intensor_[gid]; }
  //inline int InternalTensorIdToJobId(int tid) const override { return intensor2job_[tid]; }
  //std::vector<int> JobIdToInternalTensorIds(const std::vector<int>& gids) const override {
    //std::vector<int> tids(gids.size());
    //for (int i = 0; i < gids.size(); i++) tids[i] = job2intensor_[gids[i]];
    //return tids;
  //}
  //std::vector<int> InternalTensorIdToJobIds(const std::vector<int>& tids) const override {
    //std::vector<int> gids(tids.size());
    //for (int i = 0; i < tids.size(); i++) gids[i] = intensor2job_[tids[i]];
    //return gids; 
  //}

 private:
  //std::vector<int> job2intensor_;
  //std::vector<int> intensor2job_;
  std::vector<std::vector<int>> execution_tracer_;
  std::vector<std::vector<std::vector<int>>> gather_tracer_;
  std::vector<std::vector<std::vector<int>>> scatter_tracer_;
};

//inline void GraphSchedulerBase::ActivateNext() {
  //++rc_;
  //if (round2offset_.size() <= rc_())
    //round2offset_.push_back(round2offset_.back() + GetJobId().size());
//}

//inline bool GraphSchedulerBase::HasChild(int job_id) const {
  //CHECK(job_id < (*children_).size());
  //return !(*children_)[job_id].empty();
//}

//inline std::vector<int> GraphSchedulerBase::CurrentRoundTensorIdsForGather(
    //const std::vector<int>& gids, int child_offset) const {
  //std::vector<int> ret;
  //if (rc_.IsForward()) {
    //for (int gid : gids) {
      //CHECK(HasChild(gid));
      //CHECK(child_offset < (*children_)[gid].size());
      //int cid = (*children_)[gid][child_offset];
      //ret.push_back(JobIdToInternalTensorIds({cid})[0]);
    //}
  //}else {
    //CHECK(child_offset == 0);
    //for (int gid : gids) {
      //CHECK((*children_)[gid].size() == 1);
      //ret.push_back(JobIdToInternalTensorIds({gid})[0]);
    //}
  //}
  //return ret;
//}

//inline std::vector<int> GraphSchedulerBase::CurrentTensorIdsForScatter(
    //const std::vector<int>& gids, int child_offset) const {
  ////CHECK(HasChild(gid));
  //std::vector<int> ret;
  //if (rc_.IsForward()) {
    //CHECK(child_offset == 0);
    //for (int gid : gids)
      //ret.push_back(JobIdToInternalTensorIds({gid})[0]);
  //}else {
    //for (int gid : gids) {
      //if ((*parents_)[gid].size() == 0) {
        //ret.push_back(JobIdToInternalTensorIds({gid})[0]);
      //}else if (child_offset < (*parents_)[gid].size()) {
        //int pid = (*parents_)[gid][child_offset];
        //ret.push_back(JobIdToInternalTensorIds({pid})[0]);
      //}else {
        //LOG(FATAL) << "Wrong config";
      //}
    //}
  //}
  //return ret;
//}

//inline int GraphSchedulerBase::GetCurrentRoundOffset() const {
  //return round2offset_[rc_()]; 
//}

} //namespace midend

#endif

