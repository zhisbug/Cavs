#include "cavs/midend/graph_scheduler.h"
#include "cavs/proto/devices.pb.h"

#include <algorithm>

using std::vector;

namespace midend {

void GraphScheduler::ActivateNext() {
  int id = GetCurrentJobId();
  CHECK(id >= 0);
  if (isForward_) {
    int pid = parent_ids_[sample_id_][id];
    if (pid != -1 && !activated_workset_[pid]) {
      pending_workset_.push_back(pid);
      activated_workset_[pid] = true;
    }
  }else {
    for (auto& cid : child_ids_[sample_id_][id]) {
      if (!activated_workset_[cid]) {
        pending_workset_.push_back(cid);
        activated_workset_[cid] = true;
      }
    }
  }
  pending_workset_.pop_front();
  //if (isForward_) {
    //if (isLeaf(id)) {
      //CHECK(id == activate_leaf_.front());
      //activate_inode_.push_back(parent_id(id));
      //activate_leaf_.pop_front();
    //}else {
      //CHECK(id == activate_inode_.front());
      //if (parent_id(id) != -1) {
        //activate_inode_.push_back(parent_id(id));
      //}
      //activate_inode_.pop_front();
    //}
  //}else {
    //if (!isLeaf(id)) {
      //CHECK(id == activate_inode_.front());
      //for (auto& i : child_id(id)) {
        //if (isLeaf(i))
          //activate_leaf_.push_back(i);
        //else
          //activate_inode_.push_back(i);
      //}
      //activate_inode_.pop_front();
    //}else {
      //CHECK(id == activate_leaf_.front());
      //activate_leaf_.pop_front();
    //}
  //}
}

int GraphScheduler::GetCurrentJobId() {
  if (!empty())
    return pending_workset_.front();
  else
    return -1;
}

//parent-idx form
int GraphScheduler::LoadGraph(const Tensor& parent_ids) {
  VLOG(V_DEBUG) << "Loading graph...";
  CHECK(parent_ids.dims() == 2) << parent_ids.debug_info();
  CHECK(parent_ids.device_type() == CPU) << parent_ids.debug_info();
  if (parent_ids_.empty() && child_ids_.empty()) {
    parent_ids_.resize(parent_ids.dims(0)); 
    child_ids_.resize(parent_ids.dims(0)); 
  }else {
    CHECK(parent_ids_.size() == parent_ids.dims(0));
    CHECK(child_ids_.size() == parent_ids.dims(0));
  }

  VLOG(V_DEBUG) << parent_ids.dims(1);
  int total_length = 0;
  for (int i = 0; i < parent_ids.dims(0); i++) {
    VLOG(V_DEBUG) << i;
    const int *start = parent_ids.data<int>() + i*parent_ids.dims(1);
    const int *end   = parent_ids.data<int>() + (i+1)*parent_ids.dims(1);
    int real_length = std::find(start, end, -1) + 1 - start;
    CHECK(real_length <= parent_ids.dims(1));
    total_length += real_length;
    VLOG(V_DEBUG) << real_length;
    parent_ids_[i].assign(start, start + real_length);
    child_ids_[i].resize(real_length);
    for (int j = 0; j < real_length-1; j++) {
      child_ids_[i][parent_ids_[i][j]].push_back(j);
    }
  }
  batch_ = parent_ids_.size();
  isForward_ = true;
  CHECK(pending_workset_.empty());
  VLOG(V_DEBUG) << "Loading graph completed...";
  return total_length;
}

void GraphScheduler::ReverseGraph() {
  CHECK(batch_ > 0);
  CHECK(parent_ids_.size() > 0);
  isForward_ = false;
  std::fill(activated_workset_.begin(), activated_workset_.end(), false);
}

void GraphScheduler::TrigerBatchId(int sample_id) {
  //CHECK(activate_leaf_.empty());
  //CHECK(activate_inode_.empty());
  CHECK(pending_workset_.empty());
  activated_workset_.resize(parent_ids_[sample_id].size(), false);
  std::fill(activated_workset_.begin(), activated_workset_.end(), false);
  CHECK(sample_id < batch_);
  if (isForward_) {
    for (int i = 0; i < parent_ids_[sample_id].size(); i++) {
      if (child_ids_[sample_id][i].empty()) {
        pending_workset_.push_back(i);
        activated_workset_[i] = true;
      }
    }
  }else {
    for (int i = parent_ids_[sample_id].size()-1; i >= 0; i--) {
      if (parent_ids_[sample_id][i] < 0) {
        pending_workset_.push_back(i);
        activated_workset_[i] = true;
      }
    }
  }
  sample_id_ = sample_id;
}

} //namespace midend
