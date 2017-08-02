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
    VLOG(V_DEBUG) << i;
    const int *start = graph_struct.data<int>() + i*max_seq_length_;
    int one_seq_length = std::find(start, start+max_seq_length_, -1) + 1 - start;
    CHECK(one_seq_length <= max_seq_length_);
    VLOG(V_DEBUG) << one_seq_length;
    total_length += one_seq_length;
    for (int j = 0; j < one_seq_length-1; j++) {
      __forward_parents_ids_[toGlobalId(i, j)].resize(1);
      __forward_parents_ids_[toGlobalId(i, j)][0] = *(start+j);
      __forward_children_ids_[toGlobalId(i, j)].clear();
    }
    __forward_children_ids_[toGlobalId(i, one_seq_length-1)].clear();
    for (int j = 0; j < one_seq_length; j++) {
      if (!__forward_parents_ids_[toGlobalId(i, j)].empty()) {
        int parent = __forward_parents_ids_[toGlobalId(i, j)][0];
        __forward_children_ids_[parent].push_back(toGlobalId(i, j));
      }
    }
  }
  //isForward_ = true;
  parents_ = &__forward_parents_ids_;
  children_ = &__forward_children_ids_;
  VLOG(V_DEBUG) << "Loading graph completed...";
  return total_length;
}

void GraphSchedulerBase::ReverseGraph() {
  CHECK(batch_size_ > 0);
  CHECK(max_seq_length_ > 0);
  //isForward_ = false;
  children_ = &__forward_parents_ids_;
  parents_ = &__forward_children_ids_;
}

void SerialGraphScheduler::Initialize() {
  CHECK(Terminate());
  std::fill(activated_ids_.begin(), activated_ids_.end(), false);
  InitializeSample(0);
  ready_to_execute_ids_[0] = pending_list_.front();
}

void SerialGraphScheduler::InitializeSample(int sid) {
  //if (isForward_) {
    //for (int i = 0; i < max_seq_length_; i++) {
      //int gid = toGlobalId(sid, i);
      //if (child_ids_[gid].empty() && !parent_ids_[gid].empty()) {
        //pending_list_.push_back(gid);
        //executed_ids_[gid] = true;
      //}
    //}
  //}else {
    //for (int i = max_seq_length_-1; i >= 0; i--) {
      //int gid = toGlobalId(sid, i);
      //if (parent_ids_[gid].empty() && !child_ids_[gid].empty()) {
        //pending_list_.push_back(gid);
        //executed_ids_[gid] = true;
      //}
    //}
  //}
  for (int i = 0; i < max_seq_length_; i++) {
    int gid = toGlobalId(sid, i);
    if ((*children_)[gid].empty() && !(*parents_)[gid].empty()) {
      pending_list_.push_back(gid);
      activated_ids_[gid] = true;
    }
  }
}

void SerialGraphScheduler::ActivateNext() {
  int gid = pending_list_.front();
  //if (isForward_) {
    //if (!parent_ids_[gid].empty()) {
      //for (int pid : parent_ids_[gid]) {
        //if (!executed_ids_[pid]) {
          //pending_list_.push_back(pid);
          //executed_ids_[pid] = true;
        //}
      //}
    //}else if (toSampleId(gid) < batch_size()-1) {
      //InitializeSample(toSampleId(gid)+1);
    //}
  //}else {
    //if (!child_ids_[gid].empty()) {
      //for (int cid : child_ids_[gid]) {
        //if (!activated_workset_[cid]) {
          //pending_list_.push_back(cid);
          //activated_workset_[cid] = true;
        //}
      //}
    //}else if (toSampleId(gid) < batch_size()-1) {
      //InitializeSample(toSampleId(gid)+1);
    //}
  //}
  if (!(*parents_)[gid].empty()) {
    for (int pid : (*parents_)[gid]) {
      if (!activated_ids_[pid]) {
        pending_list_.push_back(pid);
        activated_ids_[pid] = true;
      }
    }
  }else if (toSampleId(gid) < batch_size()-1) {
    InitializeSample(toSampleId(gid)+1);
  }
  pending_list_.pop_front();
  ready_to_execute_ids_[0] = pending_list_.front();
}

void BatchGraphScheduler::Initialize() {
  job2tensor_.resize(activated_ids_.size());
  for (int sid = 0; sid < batch_size(); sid++) {
    for (int i = 0; i < max_seq_length_; i++) {
      int gid = toGlobalId(sid, i);
      if ((*children_)[gid].empty() && !(*parents_)[gid].empty()) {
        activated_ids_[gid] = true;
        ready_to_execute_ids_.push_back(gid);
        job2tensor_[gid] = executed_jobs_ + (ready_to_execute_ids_.size()-1);
      }
    }
  }
}

void BatchGraphScheduler::ActivateNext() {
  vector<int> jobs_next_round;
  for (int gid : ready_to_execute_ids_) {
    for (int pid : (*parents_)[gid]) {
      if (!activated_ids_[pid]) {
        activated_ids_[pid] = true;
        jobs_next_round.push_back(pid);
      }
    }
  }
  executed_jobs_ += ready_to_execute_ids_.size();
  ready_to_execute_ids_ = std::move(jobs_next_round);
  if (ready_to_execute_ids_.empty())
    executed_jobs_ = 0;
}

} //namespace midend
