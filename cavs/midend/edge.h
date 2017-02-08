#ifndef CAVS_MIDEND_EDGE_H_
#define CAVS_MIDEND_EDGE_H_

#include "cavs/midend/scope.h"
#include "cavs/midend/node.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>
#include <algorithm>

namespace midend {

class Scope;
class Node;
extern const Scope* GetGlobalScope();

class Edge {
 public:
  explicit Edge(const std::string& name, bool stateful, 
      const Scope* s = GetGlobalScope())
    : tensor_name_(name), stateful_(stateful),
      s_(s), sink_(NULL) {} 
  bool isStateful() const { return stateful_; }
  inline const std::string& name() const {
    return tensor_name_;
  }
  inline void SetShape(const TensorShapeDef& def) {
    tensor_shape_ = def;  
  }
  inline const TensorShapeDef& shape() const {
    return tensor_shape_; 
  }
  inline void AddSource(Node* node) {
    CHECK(!stateful_ || srcs_.empty());
    srcs_.push_back(node); 
  }
  void AddDst(Node* node);
  inline void RemoveDst(Node* node) {
    std::remove(dsts_.begin(), dsts_.end(), node); 
  }
  inline const Node* src(size_t i) const {
    CHECK(i < srcs_.size());
    return srcs_[i];
  }
  inline const std::vector<Node*>& srcs() const {
    return srcs_;
  }
  inline int srcs_size() const {
    return srcs_.size();
  }
  inline const Node* dst(size_t i) const {
    CHECK(i < dsts_.size());
    return dsts_[i];
  }
  inline const std::vector<Node*>& dsts() const {
    return dsts_;
  }
  inline int dsts_size() const {
    return dsts_.size();
  }

 private:
  Node* sink_;
  std::string tensor_name_;
  TensorShapeDef tensor_shape_;
  std::vector<Node*> srcs_;
  std::vector<Node*> dsts_;
  bool stateful_;
  const Scope* s_;
};

} //namespace midend

#endif
