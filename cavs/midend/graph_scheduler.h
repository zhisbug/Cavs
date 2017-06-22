#ifndef CAVS_MIDEND_GRAPH_SCHEDULER_H_
#define CAVS_MIDEND_GRAPH_SCHEDULER_H_

#include <vector>

class GraphScheduler {
 public:
  static void ActiveNext(int id) {

  }
  static void GetLeafJob() {

  }
  static void GetInodeJob() {

  }
  static bool LeafEmpty() {
    return Get()->activate_leaf_.empty(); 
  }
  static bool InodeEmpty() {
    return Get()->activate_inode.empty(); 
  }

 private:
  GraphScheduler() {}
  static GraphScheduler* Get() { GraphScheduler gs; return &gs; }
  std::vector<int> activate_leaf_;
  std::vector<int> activate_inode_;
  std::vector<int> parent_ids_;
};

class BatchedGraphScheduler : public GraphScheduler {

};

#endif

