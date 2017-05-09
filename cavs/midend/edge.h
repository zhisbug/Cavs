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
extern Scope* GetGlobalScope();

class Edge {
 public:
  explicit Edge(const std::string& name, bool stateful, 
      Scope* s = GetGlobalScope());
  inline bool isStateful() const;
  inline bool isVirtual() const;
  inline const std::string& name() const;
  inline const std::string& scoped_name() const;
  inline const Scope* scope() const;
  inline void SetShape(const TensorShapeDef& def);
  inline const TensorShapeDef& shape() const;
  inline void RemoveDst(Node* node);
  inline const Node* src(size_t i) const;
  inline const std::vector<Node*>& srcs() const;
  inline int srcs_size() const;
  inline const Node* dst(size_t i) const;
  inline const std::vector<Node*>& dsts() const;
  inline int dsts_size() const;
  void AddSource(Node* node);
  void AddDst(Node* node);
  std::string DebugInfo() const;

 private:
  std::string name_;
  std::string scoped_name_;
  TensorShapeDef tensor_shape_;
  std::vector<Node*> srcs_;
  std::vector<Node*> dsts_;
  bool stateful_;
  Scope* s_;
};

inline bool Edge::isStateful() const {
  return stateful_; 
}

inline bool Edge::isVirtual() const {
  return tensor_shape_.dim_size() == 0; 
}

inline const std::string& Edge::name() const {
  return name_;
}

inline const std::string& Edge::scoped_name() const {
  return scoped_name_;
}

inline const Scope* Edge::scope() const {
  return s_; 
}

inline void Edge::SetShape(const TensorShapeDef& def) {
  tensor_shape_ = def;  
}

inline const TensorShapeDef& Edge::shape() const {
  return tensor_shape_; 
}

inline void Edge::RemoveDst(Node* node) {
  std::remove(dsts_.begin(), dsts_.end(), node); 
}

inline const Node* Edge::src(size_t i) const {
  CHECK(i < srcs_.size());
  return srcs_[i];
}

inline const std::vector<Node*>& Edge::srcs() const {
  return srcs_;
}

inline int Edge::srcs_size() const {
  return srcs_.size();
}

inline const Node* Edge::dst(size_t i) const {
  CHECK(i < dsts_.size())
       << i << "\t" << dsts_size()
       << DebugInfo();
  return dsts_[i];
}

inline const std::vector<Node*>& Edge::dsts() const {
  return dsts_;
}

inline int Edge::dsts_size() const {
  return dsts_.size();
}

} //namespace midend

#endif
