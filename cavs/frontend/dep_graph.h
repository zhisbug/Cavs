#ifndef CAVS_FRONTEND_DEP_GRAPH_H_
#define CAVS_FRONTEND_DEP_GRAPH_H_

#include "cavs/proto/op_def.pb.h"

#include <vector>
#include <unordered_map>
#include <string>

namespace frontend {

class Node;
class Edge;
class DepGraph {
 public:
  Node* AddNode(const OpDef& op_def);
  inline int num_nodes() const {
    return nodes_.size();
  }
  inline const Node* operator[](int node_id) const {
    return nodes_[node_id];
  }
  void GradientPass();

 private:
  std::vector<Node*> nodes_;
  std::vector<Edge*> edges_;
  std::unordered_map<std::string, Edge*> out2edge_;
};

class Node {
 public:
  explicit Node(const OpDef& op_def) : op_def_(op_def) {}
  inline void AddInput(Edge* e) {
    inputs_.push_back(e);
  }
  inline void AddOutput(Edge* e) {
    outputs_.push_back(e);
  }
  void InputShapes(std::vector<TensorShapeDef>* inputs);
  inline const OpDef& op_def() const {
    return op_def_; 
  }
  inline void SetShape(const std::vector<TensorShapeDef>& def);

 private:
  OpDef op_def_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;
};

class Edge {
 public:
  explicit Edge(const std::string& name) : tensor_name_(name) {} 
  inline void SetShape(const TensorShapeDef& def) {
    tensor_shape_ = def;  
  }
  inline const TensorShapeDef& shape() const {
    return tensor_shape_; 
  }
  inline void SetSource(const Node* node) {
    src_ = const_cast<Node*>(node);
  }
  inline void AddDst(const Node* node) {
    dst_.push_back(node); 
  }

 private:
  std::string tensor_name_;
  TensorShapeDef tensor_shape_;
  Node* src_;
  std::vector<const Node*> dst_;
};

inline void Node::SetShape(const std::vector<TensorShapeDef>& def) {
  for (int i = 0; i < outputs_.size(); i++) {
    outputs_[i]->SetShape(def[i]);
  }
}

} //namespace frontend

#endif
