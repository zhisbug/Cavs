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
class SessionBase;

class Node {
 public:
  virtual Statement* Compile(SessionBase* sess) {
    return NULL;
  }
  virtual bool IsSingleNode() const { return false; }
  virtual bool IsScopedNode() const { return false; }
  inline const OpDef& op_def() const;
  inline Scope*                    scope()         const;
  inline Edge*                     input(int idx)  const;
  inline const std::vector<Edge*>& input()         const;
  inline int                       input_size()    const;
  inline Edge*                     output(int idx) const;
  inline const std::vector<Edge*>& output()        const;
  inline int                       output_size()   const;

  std::string name()        const;
  std::string scoped_name() const;
  std::vector<TensorShapeDef> input_shapes();

  void AddInput(const Edge* e);
  void AddOutput(const Edge* e);
  //inline void RemoveInput(const Edge* e);
  //inline void replaceInput(int i, Edge* edge);
  void SetShape(const std::vector<TensorShapeDef>& def);

  std::string debug_info() const;

 protected:
  explicit Node(const OpDef& op_def, Scope* s);
  OpDef op_def_;
  Scope* located_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;
  std::string node_name_;
  Statement* stmt_;
};

class SingleNode : public Node {
 public:
  explicit SingleNode(const OpDef& op_def, Scope* s)
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
  explicit ScopedNode(
      Scope* located, 
      const Scope* contained,
      const OpDef& op_def,
      int iter);
  Statement* Compile(SessionBase* sess) override;
  bool IsScopedNode() const override { return true; }
  std::list<Node*> nodes_;

 private:
  int iter_;
  const Scope* contained_;
  void Compress();
};

inline const OpDef& Node::op_def() const {
  return op_def_;
}

inline Scope* Node::scope() const {
  return located_;
}


inline Edge* Node::input(int idx) const {
  CHECK(idx < inputs_.size());
  return inputs_[idx];
}

inline const std::vector<Edge*>& Node::input() const {
  return inputs_; 
}

inline int Node::input_size() const {
  return inputs_.size();
}

inline Edge* Node::output(int idx) const {
  CHECK(idx < outputs_.size())
        << debug_info()
        << "\nAcquiring idx: " << idx
        << "\nSize: " << output_size();
  return outputs_[idx];
}

inline const std::vector<Edge*>& Node::output() const {
  return outputs_;
}

inline int Node::output_size() const {
  return outputs_.size();
}

//inline void Node::RemoveInput(const Edge* e) {
  //std::remove(inputs_.begin(), inputs_.end(), e);  
//}

//inline void Node::replaceInput(int i, Edge* edge) {
  //CHECK(i < inputs_.size());
  //inputs_[i] = edge; 
//}

} //namespace midend

#endif
