#include "cavs/midend/graph_scheduler.h"

using std::vector;

namespace midend {

void GraphScheduler::ActiveItsParent(int id) {
  if (isLeaf(id)) {
    CHECK(id == Get()->activate_leaf_.front());
    Get()->activate_inode_.push_back(parent_id(id));
  }else {
    CHECK(id == Get()->activate_inode_.front());
    if (parent_id(id) != -1) {
      Get()->activate_inode_.push_back(parent_id(id));
    }
  }
}

int GraphScheduler::GetJobId() {
  if (!Get()->activate_leaf_.empty()) {
    return Get()->activate_leaf_.front();
  }else if (!Get()->activate_inode_.empty()) {
    return Get()->activate_inode_.front();
  }else {
    LOG(FATAL) << "Non job left";
  }
}

//parent-idx form
void GraphScheduler::LoadGraph(vector<int>&& parent_ids) {
  Get()->parent_ids_ = std::move(parent_ids);
  Get()->child_ids_.resize(parent_ids.size());
  for (int i = 0; i < parent_ids.size(); i++) {
    CHECK(parent_ids[i] < parent_ids.size());
    Get()->child_ids_[parent_ids[i]].push_back(i);
    if (Get()->child_ids_[i].empty()) {//must be the leaf node
      Get()->activate_leaf_.push_back(i); 
    }
  }
}

} //namespace midend
