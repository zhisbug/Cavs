#include "cavs/midend/graph_scheduler.h"
#include "cavs/proto/devices.pb.h"

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
    activated_ids_.resize(batch_size_*max_seq_length_, false);
  }else {
    CHECK(batch_size_ == graph_struct.dims(0)); 
    CHECK(max_seq_length_ == graph_struct.dims(1)); 
  }

  int total_length = 0;
  for (int i = 0; i < batch_size_; i++) {
    const int *start = graph_struct.data<int>() + i*max_seq_length_;
    int one_seq_length = std::find(start, start+max_seq_length_, -1) + 1 - start;
    CHECK(one_seq_length <= max_seq_length_);
    total_length += one_seq_length;
    for (int j = 0; j < one_seq_length-1; j++) {
      __forward_parents_ids_[toGlobalId(i, j)].resize(1);
      __forward_parents_ids_[toGlobalId(i, j)][0] = toGlobalId(i, *(start+j));
      __forward_children_ids_[toGlobalId(i, j)].clear();
      VLOG(V_DEBUG) << "parents: " << i << "\t" << j << "\t" << toGlobalId(i, j)
                    << "\t" << __forward_parents_ids_[toGlobalId(i, j)][0];
    }
    __forward_children_ids_[toGlobalId(i, one_seq_length-1)].clear();
    for (int j = 0; j < one_seq_length; j++) {
      if (!__forward_parents_ids_[toGlobalId(i, j)].empty()) {
        int parent = __forward_parents_ids_[toGlobalId(i, j)][0];
        __forward_children_ids_[parent].push_back(toGlobalId(i, j));
        VLOG(V_DEBUG) << "children: " << i << "\t" << j << "\t" << parent
                      << "\t" << __forward_children_ids_[parent].back();
      }
    }
  }
  //isForward_ = true;
  parents_ = &__forward_parents_ids_;
  children_ = &__forward_children_ids_;
  round2offset_.clear();
  rc_.SetForward();
  CHECK(rc_() == -1) << rc_();
  ++rc_;
  round2offset_.push_back(0);
  VLOG(V_DEBUG) << "Loading graph completed...";
  return total_length;
}

void GraphSchedulerBase::ReverseGraph() {
  CHECK(batch_size_ > 0);
  CHECK(max_seq_length_ > 0);
  //isForward_ = false;
  children_ = &__forward_parents_ids_;
  parents_ = &__forward_children_ids_;
  rc_.SetBackward();
  ++rc_;
}

void SerialGraphScheduler::Initialize() {
  CHECK(Terminate());
  std::fill(activated_ids_.begin(), activated_ids_.end(), false);
  InitializeSample(0);
  ready_to_execute_ids_[0] = pending_list_.front();
}

void SerialGraphScheduler::InitializeSample(int sid) {
  for (int i = 0; i < max_seq_length_; i++) {
    int gid = toGlobalId(sid, i);
    if ((*children_)[gid].empty() && !(*parents_)[gid].empty()) {
      pending_list_.push_back(gid);
      activated_ids_[gid] = true;
      VLOG(V_DEBUG) << "Activating job_id: " << gid;
    }
  }
}

void SerialGraphScheduler::ActivateNext() {
  int gid = pending_list_.front();
  GraphSchedulerBase::ActivateNext();
  if (!(*parents_)[gid].empty()) {
    for (int pid : (*parents_)[gid]) {
      if (!activated_ids_[pid]) {
        pending_list_.push_back(pid);
        activated_ids_[pid] = true;
        VLOG(V_DEBUG) << "Activating job_id: " << pid;
      }
    }
  }else if (toSampleId(gid) < batch_size()-1) {
    InitializeSample(toSampleId(gid)+1);
  }
  pending_list_.pop_front();
  ready_to_execute_ids_[0] = pending_list_.front();
}

void BatchGraphScheduler::Initialize() {
  CHECK(ready_to_execute_ids_.empty());
  if (rc_.IsForward()) {
    for (int sid = 0; sid < batch_size(); sid++) {
      for (int i = 0; i < max_seq_length_; i++) {
        int gid = toGlobalId(sid, i);
        if ((*children_)[gid].empty() && !(*parents_)[gid].empty()) {
          activated_ids_[gid] = true;
          job2intensor_[gid] = GetCurrentRoundOffset() + ready_to_execute_ids_.size();
          ready_to_execute_ids_.push_back(gid);
        }
      }
    }
    execution_tracer_.clear();
  }else {
    ready_to_execute_ids_ = std::move(execution_tracer_[rc_()]);
  }
}

void BatchGraphScheduler::ActivateNext() {
  GraphSchedulerBase::ActivateNext();
  if (rc_.IsForward()) {
    vector<int> jobs_next_round;
    for (int gid : ready_to_execute_ids_) {
      for (int pid : (*parents_)[gid]) {
        if (!activated_ids_[pid]) {
          activated_ids_[pid] = true;
          job2intensor_[gid] = GetCurrentRoundOffset() + jobs_next_round.size();
          jobs_next_round.push_back(pid);
        }
      }
    }
    execution_tracer_.push_back(std::move(ready_to_execute_ids_));
    ready_to_execute_ids_ = std::move(jobs_next_round);
  }else {
    ready_to_execute_ids_ = std::move(execution_tracer_[rc_.prev()]);
  }
}

} //namespace midend
