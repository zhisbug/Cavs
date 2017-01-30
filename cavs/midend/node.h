#ifndef CAVS_MIDEND_NODE_H_
#define CAVS_MIDEND_NODE_H_

#include "cavs/midend/scope.h"
#include "cavs/midend/edge.h"
#include "cavs/proto/op_def.pb.h"

#include <string>
#include <vector>

class Node {
 public:
  explicit Node(const OpDef& op_def, const Scope* s)
    : op_def_(op_def), s_(s) {}
  inline void AddInput(const Edge* e) {
    inputs_.push_back(const_cast<Edge*>(e));
  }
  inline void AddOutput(const Edge* e) {
    outputs_.push_back(const_cast<Edge*>(e));
  }
  inline void InputShapes(std::vector<TensorShapeDef>* inputs);
  //inline const OpDef& op_def() const {
    //return op_def_; 
  //}
  std::string& name() const { return op_def_.name(); }
  inline void SetShape(const std::vector<TensorShapeDef>& def);
  inline bool IsVariableOp() const {
    return (op_def_.name() == "Variable");
  }
  inline const Edge* inputs(int i) {
    CHECK(i < inputs_.size());
    return inputs_[i];
  }
  inline int inputs_size() const {
    return inputs_.size();
  }
  inline const Edge* outputs(int i) {
    CHECK(i < outputs_.size());
    return outputs_[i];
  }
  inline int outputs_size() const {
    return outputs_.size();
  }
  //friend class DepGraph;
  //friend class NodeGroup;

 private:
  OpDef op_def_;
  Scope* s_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;
};

inline void Node::SetShape(
    const std::vector<TensorShapeDef>& def) {
  op_def_.clear_shape();
  for (int i = 0; i < outputs_.size(); i++) {
    outputs_[i]->SetShape(def[i]);
    *(op_def_.add_shape()) = def[i];
  }
}

inline void Node::InputShapes(
    vector<TensorShapeDef>* inputs) {
  for (auto* edge: inputs_) {
    inputs->push_back(edge->shape());
  }
}


#endif
