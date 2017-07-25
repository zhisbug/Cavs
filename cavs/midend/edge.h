#ifndef CAVS_MIDEND_EDGE_H_
#define CAVS_MIDEND_EDGE_H_

#include "cavs/midend/scope.h"
#include "cavs/midend/node.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/logging.h"
#include "cavs/util/op_util.h"

#include <string>
#include <vector>
#include <algorithm>

namespace midend {

class Scope;
class Node;

class Edge {
 public:
  explicit Edge(const std::string& name, Scope* s);
  inline bool isVariable() const;
  inline bool isVirtual() const;
  inline bool isGradient() const;
  inline std::string name() const;
  inline Scope* scope() const;
  std::string scoped_name() const;

  inline Node*                     src(int idx, bool within=false) const;
  inline const std::vector<Node*>& src(bool within=false)          const;
  inline int                       src_size(bool within=false)     const;
  inline Node*                     dst(int idx, bool within=false) const;
  inline const std::vector<Node*>& dst(bool within=false)          const;
  inline int                       dst_size(bool within=false)     const;
  inline const std::vector<Node*>& control_dependency()            const;

  void AddSource(Node* node);
  void AddDst(Node* node);
  void AddControlDependency(const Node* n);

  inline void SetShape(const TensorShapeDef& def);
  inline const TensorShapeDef& shape() const;
  DataType dtype() const;

  std::string debug_info() const;

 private:
  std::string name_;
  TensorShapeDef tensor_shape_;
  std::vector<Node*> srcs_;
  std::vector<Node*> same_scoped_srcs_;
  std::vector<Node*> dsts_;
  std::vector<Node*> same_scoped_dsts_;
  std::vector<Node*> control_dependency_on_me_;
  Scope* located_;
};

inline bool Edge::isVariable() const {
  return IsVariableName(name_); 
}

inline bool Edge::isGradient() const {
  return IsGradientName(name_); 
}

inline bool Edge::isVirtual() const {
  return tensor_shape_.dim_size() == 0; 
}

inline std::string Edge::name() const {
  return name_;
}

//inline const std::string& Edge::scoped_name() const {
  //return scoped_name_;
//}

inline Scope* Edge::scope() const {
  return located_; 
}

inline void Edge::SetShape(const TensorShapeDef& def) {
  tensor_shape_ = def;  
}

inline const TensorShapeDef& Edge::shape() const {
  return tensor_shape_; 
}

inline Node* Edge::src(int idx, bool within) const {
  CHECK(idx < src_size(within));
  if (!within)
    return srcs_[idx];
  else
    return same_scoped_srcs_[idx];
}

inline const std::vector<Node*>& Edge::src(bool within) const {
  if (!within)
    return srcs_;
  else
    return same_scoped_srcs_;
}

inline int Edge::src_size(bool within) const {
  if (!within)
    return srcs_.size();
  else
    return same_scoped_srcs_.size();
}

inline Node* Edge::dst(int idx, bool within) const {
  CHECK(idx < dst_size(within))
       << idx << "\t" << dst_size(within)
       << debug_info();
  if (!within)
    return dsts_[idx];
  else
    return same_scoped_dsts_[idx];
}

inline const std::vector<Node*>& Edge::dst(bool within) const {
  if (!within)
    return dsts_;
  else
    return same_scoped_dsts_;
}

inline int Edge::dst_size(bool within) const {
  if (!within)
    return dsts_.size();
  else
    return same_scoped_dsts_.size();
}

inline const std::vector<Node*>& Edge::control_dependency() const {
  return control_dependency_on_me_;
}

} //namespace midend

#endif
