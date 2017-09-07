#ifndef CAVS_MIDEND_NODE_H_
#define CAVS_MIDEND_NODE_H_

#include "cavs/midend/scope.h"
#include "cavs/midend/edge.h"
#include "cavs/midend/session_base.h"
#include "cavs/midend/statement.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/logging.h"
#include "cavs/util/op_util.h"

#include <string>
#include <vector>
#include <list>
#include <algorithm>

namespace midend {

class Scope;
class Edge;
class SessionBase;
class GraphSession;

class Node {
 public:
  virtual Statement* Compile(SessionBase* sess) = 0;
  virtual bool IsSingleNode()  const { return false; }
  virtual bool IsScopedNode()  const { return false; }

  inline Scope* scope() const { return located_; }
  inline Edge*                     input(int idx)       const;
  inline const std::vector<Edge*>& input()              const;
  inline int                       input_size()         const;
  inline Edge*                     output(int idx)      const;
  inline const std::vector<Edge*>& output()             const;
  inline int                       output_size()        const;
  inline const std::vector<Edge*>& control_dependency() const;
  inline bool                      IsStatefulOp()       const;
  std::vector<TensorShapeDef>      input_shapes()       const;

  void AddInput(const Edge* e);
  void AddOutput(const Edge* e);
  void AddControlDependency(const Edge* e);

  virtual std::string name()   const = 0;
  std::string scoped_name()    const;
  virtual std::string debug_info() const;

 protected:
  explicit Node(Scope* located);
  OpDef op_def_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;
  std::vector<Edge*> control_dependency_;
  Scope* located_;
  Statement* stmt_;
};

class SingleNode : public Node {
 public:
  SingleNode(const OpDef& op_def, Scope* s);
  Statement* Compile(SessionBase* sess) override;
  void SetShape(const std::vector<TensorShapeDef>& def);
  void SetDynamicEnabled();

  inline bool IsSingleNode() const override { return true;  }
  inline bool IsVariableOp() const {
    return (op_def_.name() == "Variable" || op_def_.name() == "DDV");
  }
  inline bool isSourceOp() const {
    return op_def_.input_size() == 0;
  }
  inline bool IsGraphOp() const { 
    return (op_def_.name() == "GraphOutput");
  }
  inline bool IsDynamicEnabled() const { 
    return isDynamicEnabled_;
  }
  inline const OpDef& op_def() const {
    return op_def_;
  }
  inline DataType dtype() const {
    return op_def_.dtype();
  }
  inline std::string name() const override {
    return op_def_.name();
  }
  std::string debug_info() const override;

 protected:
  OpDef op_def_;
 private:
  SessionBase* sess_debug_;
  bool isDynamicEnabled_;
}; 

class GraphNode : public SingleNode {
 public:
  GraphNode(const OpDef& op_def, Scope* s);
  Statement* Compile(SessionBase* sess) override;
  //friend class GraphGradNode;

 private:
  GraphSession* gsess_;
}; 

class GraphGradNode : public SingleNode {
 public:
  GraphGradNode(const OpDef& op_def, Scope* s);
    //: GraphNode(op_def, s)[>, forward_node_(NULL)<] {}
  Statement* Compile(SessionBase* sess) override;
  //void SetGraphForwardNode(GraphNode* n) {
    //forward_node_ = n; 
  //}
 private:
  GraphSession* gsess_;
}; 

//The ScopedNode is defined as a group of nodes
//which update some certain tensors iteratively.
//Motivated by how to denote embedded loops in the 
//computation graph, the ScopedNode is introduced
//and will be translated into a basic block during
//compilation
class ScopedNode : public Node {
 public:
  ScopedNode(Scope* located, const std::string& name, int iter);
  void SetContainedScope(const Scope* contained);
  Statement* Compile(SessionBase* sess) override;
  inline bool IsScopedNode() const override { return true; }
  inline std::string name() const override {
    return name_;
  }
  std::string debug_info() const override;
  std::list<Node*> nodes_;

 private:
  std::string name_;
  int iter_;
  const Scope* contained_;
};

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

inline const std::vector<Edge*>& Node::control_dependency() const {
  return control_dependency_;
}

inline bool Node::IsStatefulOp() const {
  return IsStatefulName(name());
}

} //namespace midend

#endif
