#include "cavs/midend/graph_scheduler.h"
#include "cavs/proto/devices.pb.h"

#include <algorithm>

using std::vector;

namespace midend {

void GraphScheduler::ActiveNext() {
  int id = GetJobId();
  if (isLeaf(id)) {
    CHECK(id == activate_leaf_.front());
    activate_inode_.push_back(parent_id(id));
    activate_leaf_.pop_front();
  }else {
    CHECK(id == activate_inode_.front());
    if (parent_id(id) != -1) {
      activate_inode_.push_back(parent_id(id));
    }
    activate_inode_.pop_front();
  }
}

int GraphScheduler::GetJobId() {
  if (!activate_leaf_.empty()) {
    return activate_leaf_.front();
  }else if (!activate_inode_.empty()) {
    return activate_inode_.front();
  }else {
    LOG(FATAL) << "Non job left";
  }
}

//parent-idx form
int GraphScheduler::LoadGraph(const Tensor& parent_ids) {
  //Get()->parent_ids_ = std::move(parent_ids);
  VLOG(V_DEBUG) << "Loading graph...";
  CHECK(parent_ids.dims() == 2) << parent_ids.debug_info();
  CHECK(parent_ids.device_type() == CPU);
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
      //if (child_ids_[i][j].empty())
        //activate_leaf_.push_back(j);
    }
  }
  sample_id_ = 0;
  batch_ = parent_ids_.size();
  CHECK(activate_leaf_.empty());
  VLOG(V_DEBUG) << "Loading graph completed...";
  return total_length;
}

void GraphScheduler::ActiveLeaf(int sample_id) {
  CHECK(activate_leaf_.empty());
  CHECK(sample_id < batch_);
  for (int i = 0; i < parent_ids_[sample_id].size(); i++) {
    if (child_ids_[sample_id][i].empty())
      activate_leaf_.push_back(i);
  }
  sample_id_ = sample_id;
}

} //namespace midend
