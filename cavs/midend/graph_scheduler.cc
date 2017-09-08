#include "cavs/midend/graph_scheduler.h"
#include "cavs/proto/devices.pb.h"
#include "cavs/util/macros_gpu.h"

#include <algorithm>

using std::vector;

namespace midend {

//parent-idx form
int GraphSchedulerBase::LoadGraph(const Tensor& graph_struct) {
  VLOG(V_DEBUG) << "Loading graph...";
  CHECK(graph_struct.dims() == 2) << graph_struct.debug_info();
  CHECK(graph_struct.device_type() == CPU) << graph_struct.debug_info();
  if (batch_size_ == 0 && max_seq_length_ == 0) {
    batch_size_ = graph_struct.dims(0);
    max_seq_length_ = graph_struct.dims(1);
    __forward_parents_ids_.resize(batch_size_*max_seq_length_); 
    __forward_children_ids_.resize(batch_size_*max_seq_length_); 
    sample_offset_in_gid_.resize(batch_size_);
    //activated_times_.resize(batch_size_*max_seq_length_, 0);
    checkCudaError(cudaMalloc((void**)&gpu_idx_buf_, batch_size_*max_seq_length_*sizeof(int)));
  }else {
    CHECK(batch_size_ == graph_struct.dims(0)); 
    CHECK(max_seq_length_ == graph_struct.dims(1)); 
  }

  total_length_ = 0;
  int prev_seq_length = 0;
  for (int i = 0; i < batch_size_; i++) {
    const int *start = graph_struct.data<int>() + i*max_seq_length_;
    int curr_seq_length = std::find(start, start+max_seq_length_, -1) + 1 - start;
    sample_offset_in_gid_[i] = prev_seq_length + ((i > 0) ? sample_offset_in_gid_[i-1] : 0);
    CHECK(curr_seq_length <= max_seq_length_);
    VLOG(V_DEBUG) << "sequence_lengh = " << curr_seq_length;
    total_length_ += curr_seq_length;
    for (int j = 0; j < curr_seq_length-1; j++) {
      __forward_parents_ids_[toGlobalId(i, j)].resize(1);
      __forward_parents_ids_[toGlobalId(i, j)][0] = toGlobalId(i, *(start+j));
      __forward_children_ids_[toGlobalId(i, j)].clear();
      VLOG(V_DEBUG) << "parents[" << i << "][" << j << "](" << toGlobalId(i, j)
                    << ") = " << __forward_parents_ids_[toGlobalId(i, j)][0];
    }
    __forward_children_ids_[toGlobalId(i, curr_seq_length-1)].clear();
    for (int j = 0; j < curr_seq_length; j++) {
      if (!__forward_parents_ids_[toGlobalId(i, j)].empty()) {
        int parent = __forward_parents_ids_[toGlobalId(i, j)][0];
        __forward_children_ids_[parent].push_back(toGlobalId(i, j));
        VLOG(V_DEBUG) << "children: " << i << "\t" << j << "\t" << parent
                      << "\t" << __forward_children_ids_[parent].back();
      }
    }
    prev_seq_length = curr_seq_length;
  }
  parents_ = &__forward_parents_ids_;
  children_ = &__forward_children_ids_;
  round2offset_.clear();
  rc_.Reset();
  activated_times_.resize(total_length_, 0);
  tids_to_jobids_.resize(total_length_, 0);
  jobids_to_tids_.resize(total_length_, 0);
  VLOG(V_DEBUG) << "Loading graph completed...";
  return total_length_;
}

int GraphSchedulerBase::ReverseGraph() {
  CHECK(batch_size_ > 0);
  //CHECK(max_seq_length_ > 0);
  children_ = &__forward_parents_ids_;
  parents_ = &__forward_children_ids_;
  rc_.SetBackward();
  //++rc_;
  return total_length_;
}

void SerialGraphScheduler::Initialize() {
  ++rc_;
  CHECK(Terminate());
  std::fill(activated_times_.begin(), activated_times_.end(), 0);
  //tids_to_jobids_.resize(activated_times_.size(), 0);
  InitializeSample(0);
  int gid = pending_list_.front();
  ready_to_execute_ids_[0] = gid;
  tids_to_jobids_[gid] = gid;

  if (rc_.IsForward()) {
    for (auto& child : tids_for_gather_)  child.clear();
    for (auto& child : tids_for_scatter_)  child.clear();
    for (int i = 0; i < (*parents_)[gid].size(); i++) {
      tids_for_scatter_[0].push_back(gid);
    }
    if (!HasChild(gid)) tids_for_gather_init_[0] = {gid};
  }else {
    for (auto& child : tids_for_gather_)  child.clear();
    for (auto& child : tids_for_scatter_)  child.clear();
    for (int i = 0; i < (*parents_)[gid].size(); i++) {
      tids_for_scatter_[i].push_back((*parents_)[gid][i]);
    }
    if (!HasChild(gid)) tids_for_gather_init_[1] = {gid};
  }
}

void SerialGraphScheduler::InitializeSample(int sid) {
  int sample_length = ((sid < batch_size()-1) ? sample_offset_in_gid_[sid+1] : total_length())
                      - sample_offset_in_gid_[sid];
  for (int i = 0; i < sample_length; i++) {
    int gid = toGlobalId(sid, i);
    if ((*children_)[gid].empty() && !(*parents_)[gid].empty()) {
      pending_list_.push_back(gid);
      VLOG(V_DEBUG) << "Activating job_id: " << gid;
    }
  }
  sample_id_ = sid;
}

void SerialGraphScheduler::ActivateNext() {
  ++rc_;
  int gid = pending_list_.front();
  if (!(*parents_)[gid].empty()) {
    for (int pid : (*parents_)[gid]) {
      if (++activated_times_[pid] == (*children_)[pid].size()) {
        pending_list_.push_back(pid);
        VLOG(V_DEBUG) << "Activating job_id: " << pid;
      }
    }
  }else if (++sample_id_ < batch_size()) {
    InitializeSample(sample_id_);
  }
  tids_to_jobids_[gid] = gid;
  pending_list_.pop_front();
  if (Terminate()) return;

  int next_gid = pending_list_.front();
  ready_to_execute_ids_[0] = next_gid;
  if (rc_.IsForward()) {
    for (auto& child : tids_for_scatter_)  child.clear();
    for (auto& child : tids_for_gather_)  child.clear();
    for (int i = 0; i < (*parents_)[next_gid].size(); i++) {
      //only a count number
      tids_for_scatter_[0].push_back(next_gid);
    }
    for (int i = 0; i < (*children_)[next_gid].size(); i++) {
      tids_for_gather_[i].push_back((*children_)[next_gid][i]);
    }
    if (!HasChild(next_gid)) tids_for_gather_init_[0] = {next_gid};
  }else {
    for (auto& child : tids_for_gather_)  child.clear();
    for (auto& child : tids_for_scatter_)  child.clear();
    for (int i = 0; i < (*children_)[next_gid].size(); i++) {
      //only a count number
      tids_for_gather_[0].push_back(next_gid);
    }
    for (int i = 0; i < (*parents_)[next_gid].size(); i++) {
      tids_for_scatter_[i].push_back((*parents_)[next_gid][i]);
    }
    if (!HasChild(next_gid)) tids_for_gather_init_[1] = {next_gid};
  }
}

void BatchGraphScheduler::Initialize() {
  ++rc_;
  CHECK(Terminate());
  std::fill(activated_times_.begin(), activated_times_.end(), 0);
  //jobids_to_tids_.resize(activated_times_.size(), 0);
  //tids_to_jobids_.resize(activated_times_.size(), 0);
  if (round2offset_.empty())  round2offset_.push_back(0);
  if (rc_.IsForward()) {
    //for (int sid = 0; sid < batch_size(); sid++) {
      //for (int i = 0; i < max_seq_length_; i++) {
    for (int gid = 0; gid < total_length(); gid++) {
      if ((*children_)[gid].empty() && !(*parents_)[gid].empty()) {
        int tensor_id = GetCurrentRoundOffset() + ready_to_execute_ids_.size();
        tids_to_jobids_[tensor_id] = gid;
        jobids_to_tids_[gid] = tensor_id;
        tids_for_scatter_[0].push_back(jobids_to_tids_[gid]);
        ready_to_execute_ids_.push_back(gid);
        VLOG(V_DEBUG) << "Pushing back " << gid;
      }
    }
    tids_for_gather_init_[0] = ready_to_execute_ids_;
    tids_for_gather_init_[1].clear();
    execution_tracer_.clear();
    gather_tracer_.clear();
    scatter_tracer_.clear();
  }else {
    ready_to_execute_ids_ = std::move(execution_tracer_[rc_()]);
    tids_for_gather_ = std::move(scatter_tracer_[rc_()]);
    tids_for_scatter_ = std::move(gather_tracer_[rc_()]);
    VLOG(V_DEBUG) << "ready_to_execute_ids_" << ready_to_execute_ids_[0];
  }
}

void BatchGraphScheduler::ActivateNext() {
  if (round2offset_.size() <= (++rc_)())
    round2offset_.push_back(round2offset_.back() + GetJobId().size());

  VLOG(V_DEBUG) << "activation next " << rc_();
  if (rc_.IsForward()) {
    vector<int> jobs_next_round;
    jobs_next_round.reserve(1<<20);
    vector<vector<int>> gather_ids_next_round(2);
    gather_ids_next_round[0].reserve(1<<20);
    gather_ids_next_round[1].reserve(1<<20);
    vector<vector<int>> scatter_ids_next_round(1);
    scatter_ids_next_round[0].reserve(1<<20);
    for (int gid : ready_to_execute_ids_) {
      for (int pid : (*parents_)[gid]) {
        if (++activated_times_[pid] == (*children_)[pid].size()) {
          //activated_times[pid]++;
          int tensor_id = GetCurrentRoundOffset() + jobs_next_round.size();
          tids_to_jobids_[tensor_id] = pid;
          jobids_to_tids_[pid] = tensor_id;
          jobs_next_round.push_back(pid);
          CHECK(gather_ids_next_round.size() >= (*children_)[pid].size());
          for (int i = 0; i < (*children_)[pid].size(); i++) {
            int cid = (*children_)[pid][i];
            gather_ids_next_round[i].push_back(jobids_to_tids_[cid]);
          }
          if ((*parents_)[pid].empty()) {
            tids_for_gather_init_[1].push_back(tensor_id);
          }else {
            for (int i = 0; i < (*parents_)[pid].size(); i++) {
              //just for counting(may be zero)
              scatter_ids_next_round[0].push_back(tensor_id);
            }
          }
        }
      }
    }
    execution_tracer_.push_back(std::move(ready_to_execute_ids_));
    gather_tracer_.push_back(std::move(tids_for_gather_));
    scatter_tracer_.push_back(std::move(tids_for_scatter_));

    ready_to_execute_ids_ = std::move(jobs_next_round);
    tids_for_gather_      = std::move(gather_ids_next_round);
    tids_for_scatter_     = std::move(scatter_ids_next_round);
  }else {
    if (rc_() >= 0) {
      ready_to_execute_ids_ = std::move(execution_tracer_[rc_()]);
      tids_for_gather_      = std::move(scatter_tracer_[rc_()]);
      tids_for_scatter_     = std::move(gather_tracer_[rc_()]);
      VLOG(V_DEBUG) << "ready_to_execute_ids_" << ready_to_execute_ids_[0];
    }else {
      ready_to_execute_ids_.clear();
      for (auto& ctidg : tids_for_gather_)  { ctidg.clear(); }
      for (auto& ctids : tids_for_scatter_) { ctids.clear(); }
    }
  }
}

} //namespace midend
