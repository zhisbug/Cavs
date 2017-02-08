#ifndef CAVS_MIDEND_NODE_H_
#define CAVS_MIDEND_NODE_H_

#include "cavs/midend/scope.h"
#include "cavs/midend/edge.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>
#include <algorithm>

namespace midend {

class Scope;
class Edge;
extern const Scope* GetGlobalScope();

class Node {
 public:
  explicit Node(const OpDef& op_def, const Scope* s = GetGlobalScope())
    : op_def_(op_def), s_(s) {}
  //explicit Node(const Scope* s = GetGlobalScope())
    //: s_(s) {}
  inline void AddInput(const Edge* e) {
    inputs_.push_back(const_cast<Edge*>(e));
  }
  inline void AddOutput(const Edge* e) {
    outputs_.push_back(const_cast<Edge*>(e));
  }
  inline void RemoveInput(const Edge* e) {
    std::remove(inputs_.begin(), inputs_.end(), e);  
  }
  inline const OpDef& op_def() const {
    return op_def_; 
  }
  inline const Scope* scope() const {
    return s_; 
  }
  inline const std::string& name() const { return op_def_.name(); }
  inline bool IsVariableOp() const {
    return (op_def_.name() == "Variable");
  }
  inline const Edge* input(int i) {
    CHECK(i < inputs_.size());
    return inputs_[i];
  }
  inline const std::vector<Edge*>& inputs() const {
    return inputs_; 
  }
  inline int inputs_size() const {
    return inputs_.size();
  }
  inline void replaceInput(int i, Edge* edge) {
    CHECK(i < inputs_.size());
    inputs_[i] = edge; 
  }
  inline const Edge* output(int i) {
    CHECK(i < outputs_.size());
    return outputs_[i];
  }
  inline const std::vector<Edge*>& outputs() const {
    return outputs_; 
  }
  inline int outputs_size() const {
    return outputs_.size();
  }
  inline void set_id(int id) { id_ = id; }
  inline int id() const { return id_; }
  inline bool isSink() const { return id_ == 0; }
  void SetShape(const std::vector<TensorShapeDef>& def);
  void InputShapes(std::vector<TensorShapeDef>* inputs);

 private:
  OpDef op_def_;
  const Scope* s_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;
  int id_;
};

} //namespace midend

#endif
