#ifndef CAVS_MIDEND_NODE_H_
#define CAVS_MIDEND_NODE_H_

#include "cavs/midend/scope.h"
#include "cavs/midend/edge.h"
#include "cavs/midend/session.h"
#include "cavs/midend/statement.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>
#include <algorithm>

namespace midend {

class Scope;
class Edge;
extern Scope* GetGlobalScope();

class Node {
 public:
  explicit Node(const OpDef& op_def,
      Scope* s = GetGlobalScope());

  virtual Statement* Compile(SessionBase* sess) const {
    return NULL; 
  }
  bool isVirtual() const {
    return op_def_.name() == "Optimizer";
  }
  inline const OpDef& op_def() const {
    return op_def_; 
  }
  inline const Scope* scope() const {
    return located_; 
  }
  inline const std::string& name() const {
    return op_def_.name();
  }
  inline const Edge* input(int i) const {
    CHECK(i < inputs_.size());
    return inputs_[i];
  }
  inline const std::vector<Edge*>& inputs() const {
    return inputs_; 
  }
  inline int inputs_size() const {
    return inputs_.size();
  }
  inline const Edge* output(int i) const {
    CHECK(i < outputs_.size());
    return outputs_[i];
  }
  inline const std::vector<Edge*>& outputs() const {
    return outputs_; 
  }
  inline int outputs_size() const {
    return outputs_.size();
  }
  inline void AddInput(const Edge* e) {
    inputs_.push_back(const_cast<Edge*>(e));
  }
  inline void AddOutput(const Edge* e) {
    outputs_.push_back(const_cast<Edge*>(e));
  }
  inline void RemoveInput(const Edge* e) {
    std::remove(inputs_.begin(), inputs_.end(), e);  
  }
  inline void replaceInput(int i, Edge* edge) {
    CHECK(i < inputs_.size());
    inputs_[i] = edge; 
  }
  void SetShape(const std::vector<TensorShapeDef>& def);
  void InputShapes(std::vector<TensorShapeDef>* inputs);
  std::string DebugInfo() const;
  //inline void set_id(int id) { id_ = id; }
  //inline int id() const { return id_; }
  //inline bool isSink() const { return id_ == 0; }

 protected:
  OpDef op_def_;
  Scope* located_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;
  //int id_;
};

class SingleNode : public Node {
 public:
  explicit SingleNode(const OpDef& op_def,
      Scope* s = GetGlobalScope())
    : Node(op_def, s) {}
  Statement* Compile(SessionBase* sess) const override;
  inline bool IsVariableOp() const {
    return (op_def_.name() == "Variable");
  }
  inline bool isSourceOp() const {
    return op_def_.input_size() == 0;
  }
}; 

class ScopedNode : public Node {
 public:
  explicit ScopedNode(int iter,
      Scope* contained,
      const OpDef& op_def = OpDef(),
      Scope* located = GetGlobalScope());
  Statement* Compile(SessionBase* sess) const override;

 private:
  int iter_;
  Scope* contained_;
  void Compress();
};

} //namespace midend

#endif
