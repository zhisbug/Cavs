#ifndef CAVS_MIDEND_GRAPH_SCHEDULER_H_
#define CAVS_MIDEND_GRAPH_SCHEDULER_H_

#include "cavs/util/logging.h"

#include <vector>
#include <list>

namespace midend {

class GraphScheduler {
 public:
  static void ActiveItsParent(int id);
  static int GetJobId();
  static bool LeafEmpty() {
    return Get()->activate_leaf_.empty(); 
  }
  static bool InodeEmpty() {
    return Get()->activate_inode_.empty(); 
  }
  static void LoadGraph(std::vector<int>&& parent_ids);
  static bool isLeaf(int id) {
    return Get()->child_ids_[id].empty();
  }
  static int parent_id(int id) {
    return Get()->parent_ids_[id]; 
  }
  static int child_id(int parent_id, int child_offset) {
    CHECK(!isLeaf(parent_id));
    CHECK(Get()->child_ids_[parent_id].size() > child_offset);
    return Get()->child_ids_[parent_id][child_offset]; 
  }
  static void* buffer(int child_id) {
    LOG(FATAL) << "How to define the unit of child";
    return (char*)Get()->__internal_storage_
         + child_id*Get()->__internal_unit_;
  }
  static void SetUnit(size_t u) {
    if (Get()->__internal_unit_ != u) {
      CHECK(Get()->__internal_unit_ == 0); 
      Get()->__internal_unit_ = u;
    }
  }

 private:
  GraphScheduler() : __internal_unit_(0) {}
  enum state {};
  static GraphScheduler* Get() { static GraphScheduler gs; return &gs; }
  std::list<int> activate_leaf_;
  std::list<int> activate_inode_;
  std::vector<int> parent_ids_;
  std::vector<std::vector<int>> child_ids_;
  void* __internal_storage_;
  size_t __internal_unit_;
};

class BatchedGraphScheduler : public GraphScheduler {

};

} //namespace midend

#endif

