#ifndef CAVS_MIDEND_DEP_GRAPH_H_
#define CAVS_MIDEND_DEP_GRAPH_H_

#include "cavs/midend/statement.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/logging.h"

#include <vector>
#include <unordered_map>
#include <string>

namespace midend {

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
  void BackPropagate(std::vector<std::string>* gen_grads,
      const std::string& loss);
  //void AddSolver(const std::string& solver, 
      //const std::vector<std::string>& var_names);
  BasicBlock* OptimizeLoss(const std::string& loss, 
      const std::string& solver, 
      const std::vector<std::string>& var_names);
  void Dump();

 private:
  std::vector<Node*> nodes_;
  std::vector<std::vector<Node*>> grad_nodes_;
  std::vector<Node*> update_nodes_;
  std::vector<Edge*> edges_;
  std::unordered_map<std::string, Edge*> out2edge_;
  void AddGradNode(const OpDef& op_def);
  void RecursiveSearchInputNode(
      const Edge* father, std::vector<Statement*>* stmts);
  void AddSolver(const std::string& solver,
      const std::vector<std::string>& var_names,
      std::vector<Statement*>* stmts);
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
  inline bool IsVariableOp() const {
    if (op_def_.name() == "Variable")
      return true;
    return false;
  }
  friend class DepGraph;

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
  inline void AddSource(const Node* node) {
    src_.push_back(node); 
  }
  inline void AddDst(const Node* node) {
    dst_.push_back(node); 
  }
  friend class DepGraph;

 private:
  std::string tensor_name_;
  TensorShapeDef tensor_shape_;
  std::vector<const Node*> src_;
  std::vector<const Node*> dst_;
};

inline void Node::SetShape(const std::vector<TensorShapeDef>& def) {
  op_def_.clear_shape();
  for (int i = 0; i < outputs_.size(); i++) {
    outputs_[i]->SetShape(def[i]);
    *(op_def_.add_shape()) = def[i];
  }
}

} //namespace midend 

#endif
