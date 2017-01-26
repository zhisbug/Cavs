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
class NodeGroup;
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
  //void AddSolver(const std::string& solver, 
      //const std::vector<std::string>& var_names);
  void GroupAllVariables(std::vector<std::string>* vars);
  void OptimizeWithLoss(const std::string& loss, 
      const std::string& solver, 
      const std::vector<std::string>& var_names);
  void Dump();

 private:
  std::vector<Node*> nodes_;
  std::vector<std::vector<Node*>> grad_nodes_;
  std::vector<Node*> update_nodes_;
  std::vector<Edge*> edges_;
  std::unordered_map<std::string, int> out2node_;
  std::unordered_map<std::string, Edge*> out2edge_;
  std::unordered_map<std::string, NodeGroup*> out2ng_;
  void AddGradNode(const OpDef& op_def);
  void SearchClosedSet(const Node* father,
      NodeGroup* ng,
      const std::vector<std::string>& vars,
      bool* contained);
  void BackPropagate();
  void AddSolver(const std::string& solver,
      const std::vector<std::string>& vars,
      std::vector<Statement*>* stmts);
  void SetLossNodeGroup(const std::string& loss,
      const std::vector<std::string>& vars);
};

class Edge {
 public:
  explicit Edge(const std::string& name, bool stateful)
    : tensor_name_(name), stateful_(stateful) {} 
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
  inline void AddSource(const Node* node) {
    CHECK(!stateful || src_.empty());
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
  bool stateful_;
};

class Node {
 public:
  explicit Node(const OpDef& op_def) : op_def_(op_def) {}
  inline void AddInput(const Edge* e) {
    inputs_.push_back(const_cast<Edge*>(e));
  }
  inline void AddOutput(const Edge* e) {
    outputs_.push_back(const_cast<Edge*>(e));
  }
  void InputShapes(std::vector<TensorShapeDef>* inputs);
  inline const OpDef& op_def() const {
    return op_def_; 
  }
  inline void SetShape(const std::vector<TensorShapeDef>& def);
  inline bool IsVariableOp() const {
    return (op_def_.name() == "Variable");
  }
  friend class DepGraph;
  friend class NodeGroup;

 private:
  OpDef op_def_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;
};

class NodeGroup {
 public:
  explicit NodeGroup(std::string name) : name_(name) {}
  void AddNode(const Node* n);
 private:
  std::vector<const Node*> nodes_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;
  std::string name_;
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
