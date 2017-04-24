#ifndef CAVS_MIDEND_NODE_H_
#define CAVS_MIDEND_NODE_H_

#include "cavs/midend/scope.h"
#include "cavs/midend/edge.h"
#include "cavs/midend/session_base.h"
#include "cavs/midend/statement.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>
#include <list>
#include <algorithm>

namespace midend {

class Scope;
class Edge;
extern Scope* GetGlobalScope();
class SessionBase;

class Node {
 public:
  explicit Node(const OpDef& op_def,
      const Scope* s = GetGlobalScope());
  virtual Statement* Compile(SessionBase* sess) {
    return NULL;
  }
  virtual bool IsSingleNode() const { return false; }
  virtual bool IsScopedNode() const { return false; }
  inline const OpDef& op_def() const;
  inline const Scope* scope() const;
  inline const std::string& name() const;
  inline const Edge* input(int i) const;
  inline const std::vector<Edge*>& inputs() const;
  inline int inputs_size() const;
  inline const Edge* output(int i) const;
  inline const std::vector<Edge*>& outputs() const;
  inline int outputs_size() const;
  inline void AddInput(const Edge* e);
  inline void AddOutput(const Edge* e);
  inline void RemoveInput(const Edge* e);
  inline void replaceInput(int i, Edge* edge);
  void SetShape(const std::vector<TensorShapeDef>& def);
  void InputShapes(std::vector<TensorShapeDef>* inputs);
  std::string DebugInfo() const;

 protected:
  OpDef op_def_;
  Scope* located_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;
  std::string node_name_;
  Statement* stmt_;
};

class SingleNode : public Node {
 public:
  explicit SingleNode(const OpDef& op_def,
      const Scope* s = GetGlobalScope())
    : Node(op_def, s) {}
  Statement* Compile(SessionBase* sess) override;
  bool IsSingleNode() const override { return true; }
  inline bool IsVariableOp() const {
    return (op_def_.name() == "Variable" || op_def_.name() == "DDV");
  }
  inline bool isSourceOp() const {
    return op_def_.input_size() == 0;
  }
}; 

//The ScopedNode is defined as a group of nodes
//which update some certain tensors iteratively.
//Motivated by how to denote embedded loops in the 
//computation graph, the ScopedNode is introduced
//and will be translated into a basic block during
//compilation
class ScopedNode : public Node {
 public:
  explicit ScopedNode(int iter,
      Scope* contained,
      const OpDef& op_def = OpDef(),
      Scope* located = GetGlobalScope());
  Statement* Compile(SessionBase* sess) override;
  bool IsScopedNode() const override { return true; }
  std::list<Node*> nodes_;

 private:
  int iter_;
  Scope* contained_;
  void Compress();
};

inline const OpDef& Node::op_def() const {
  return op_def_;
}
inline const Scope* Node::scope() const {
  return located_;
}
inline const std::string& Node::name() const {
  return node_name_;
}
inline const Edge* Node::input(int i) const {
  CHECK(i < inputs_.size());
  return inputs_[i];
}
inline const std::vector<Edge*>& Node::inputs() const {
  return inputs_; 
}
inline int Node::inputs_size() const {
  return inputs_.size();
}
inline const Edge* Node::output(int i) const {
  CHECK(i < outputs_.size())
    << DebugInfo()
    << "\nAcquiring idx: " << i
    << "\nSize: " << outputs_.size();
  return outputs_[i];
}
inline const std::vector<Edge*>& Node::outputs() const {
  return outputs_;
}
inline int Node::outputs_size() const {
  return outputs_.size();
}
inline void Node::AddInput(const Edge* e) {
  inputs_.push_back(const_cast<Edge*>(e));
}
inline void Node::AddOutput(const Edge* e) {
  outputs_.push_back(const_cast<Edge*>(e));
}
inline void Node::RemoveInput(const Edge* e) {
  std::remove(inputs_.begin(), inputs_.end(), e);  
}
inline void Node::replaceInput(int i, Edge* edge) {
  CHECK(i < inputs_.size());
  inputs_[i] = edge; 
}

} //namespace midend

#endif
