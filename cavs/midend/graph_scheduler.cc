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
void GraphScheduler::LoadGraph(const Tensor& parent_ids) {
  //Get()->parent_ids_ = std::move(parent_ids);
  CHECK(parent_ids.dims() == 1);
  Get()->parent_ids_.assign(parent_ids.data<int>(),
                            parent_ids.data<int>() + parent_ids.count());
  Get()->child_ids_.resize(Get()->parent_ids_.size());
  for (int i = 0; i < Get()->parent_ids_.size(); i++) {
    CHECK(Get()->parent_ids_[i] < Get()->parent_ids_.size());
    Get()->child_ids_[Get()->parent_ids_[i]].push_back(i);
    if (Get()->child_ids_[i].empty()) {//must be the leaf node
      Get()->activate_leaf_.push_back(i);
    }
  }
}

} //namespace midend
