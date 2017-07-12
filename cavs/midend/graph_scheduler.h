#ifndef CAVS_MIDEND_GRAPH_SCHEDULER_H_
#define CAVS_MIDEND_GRAPH_SCHEDULER_H_

#include "cavs/midend/tensor.h"
#include "cavs/util/logging.h"

#include <vector>
#include <list>

namespace midend {

class GraphScheduler {
 public:
  GraphScheduler() : parent_ids_(0), batch_(0), sample_id_(0), isForward_(true) {}
  inline bool LeafEmpty() const { return activate_leaf_.empty(); } 
  inline bool InodeEmpty() const { return activate_inode_.empty(); }
  void ActiveFirstWorkset(int sample_id);
  void ActiveNext();
  int LoadGraph(const Tensor& parent_ids);
  void ReverseGraph();
  int GetJobId();
  inline int batch() const { return batch_; }

  inline void SetMessagePasser(const Tensor* t) { message_passer_ = *t; }
  inline const Tensor& GetMessagePasser(int id) {
    message_passer_.SetOffsetWithId(id);
    return message_passer_; 
  }
  inline void SetMessagePusher(const Tensor* t) { message_pusher_ = *t; }
  inline const Tensor& GetMessagePusher() {
    message_pusher_.SetOffsetWithId(0);
    return message_pusher_; 
  }

  bool isLeaf(int id) const {
    CHECK(sample_id_ < child_ids_.size());
    CHECK(id < child_ids_[sample_id_].size());
    return child_ids_[sample_id_][id].empty();
  }
  int parent_id(int id) const {
    CHECK(sample_id_ < parent_ids_.size());
    CHECK(id < parent_ids_[sample_id_].size());
    return parent_ids_[sample_id_][id]; 
  }
  const std::vector<int>& child_id(int parent_id) const {
    CHECK(sample_id_ < child_ids_.size());
    CHECK(!isLeaf(parent_id));
    return child_ids_[sample_id_][parent_id]; 
  }

 private:
  //static GraphScheduler* Get() { static GraphScheduler gs; return &gs; }
  std::list<int> activate_leaf_;
  std::list<int> activate_inode_;
  std::vector<std::vector<int>> parent_ids_;
  std::vector<std::vector<std::vector<int>>> child_ids_;
  int sample_id_;
  int batch_;
  bool isForward_;
  Tensor message_passer_;
  Tensor message_pusher_;
};

class BatchedGraphScheduler : public GraphScheduler {

};

} //namespace midend

#endif

