#include "scope.h"
#include "cavs/backend/op_decl.h"

#include <vector>

using std::string;
using std::vector;

namespace midend {

Scope::Scope(const Scope* s, const std::string& n)
    : father_(s), name_(n) {
  if (s) {
    CHECK(s->children_.find(n) == s->children_.end());
    const_cast<Scope*>(s)->children_[n] = const_cast<Scope*>(this);
  }
}

Scope* Scope::FindChild(const string& n) const {
  if (children_.find(n) == children_.end())
    return NULL;
  else
    return const_cast<Scope*>(this)->children_[n]; 
}

//Node* Scope::FindNode(const string& n, bool within) const {
  //const Scope* s = this;
  //if (within) {
    //if (s->node_table_.find(n) == s->node_table_.end())
      //return NULL;
    //else
      //return s->node_table_.at(n);
  //}else {
    //while (s && s->node_table_.find(n) == s->node_table_.end()) {
      //s = s->father_; 
    //}
    //if (s)
      //return s->node_table_.at(n);
    //else
      //return NULL;
  //}
//}

Edge* Scope::FindEdge(const string& n, bool within) const {
  const Scope* s = this;
  if (within) {
    if (s->edge_table_.find(n) == s->edge_table_.end())
      return NULL;
    else
      return s->edge_table_.at(n);
  }else {
    while (s && s->edge_table_.find(n) == s->edge_table_.end()) {
      s = s->father_;
    }
    if (s)
      return s->edge_table_.at(n);
    else
      return NULL;
  }
}

Node* Scope::AddNode(const OpDef& op_def) {
  Node* node = new Node(op_def);
  nodes_.push_back(node);
  //node->set_id(nodes_.size());
  //PrintSymbolTable();
  for (auto& out : op_def.output()) {
    Edge* upper_out_edge = FindEdge(out, false);
    Edge* out_edge = FindEdge(out, true);
    bool stateful = (op_def.name() == "Variable");
    //currently, only the variables can cross scopes
    if (stateful) {
      if (!upper_out_edge) {
        CHECK(!father_);
        out_edge = new Edge(out, stateful, this);
        const_cast<Scope*>(this)->AddEdge(out_edge);
        out_edge->AddSource(node);
        node->AddOutput(out_edge);
      }else {
        CHECK(father_ && !out_edge);
        upper_out_edge->AddSource(node);
        node->AddOutput(upper_out_edge);
      }
    }else {
      if (!out_edge) {
        out_edge = new Edge(out, stateful, this);
        const_cast<Scope*>(this)->AddEdge(out_edge);
      }
      out_edge->AddSource(node);
      node->AddOutput(out_edge);
      if (upper_out_edge) {
        for (int i = 0; i < upper_out_edge->dsts_size(); i++) {
          if (upper_out_edge->dst(i)->scope() == this) {
            const_cast<Node*>(upper_out_edge->dst(i))->replaceInput(i, out_edge);
          }
        }
      }
    }
  }
  for (auto& input : op_def.input()) {
    const Edge* in_edge = FindEdge(input);
    CHECK(in_edge);
    node->AddInput(in_edge);
    const_cast<Edge*>(in_edge)->AddDst(node);
    if (edge_table_.find(in_edge->name()) ==
        edge_table_.end()) {
      in_edges_[in_edge->name()] = const_cast<Edge*>(in_edge);
    }
  }
  return node;
}

void Scope::AddGradNode(const OpDef& op_def) {
  const vector<OpDef>& grads = 
    ::backend::MakeGradient(op_def); 
  //vector<Node*> grad_vec;
  for (auto& grad : grads) {
    Node* node = new Node(grad);
    nodes_.push_back(node);
    //grad_vec.push_back(node);
    for (auto& dx : op_def.output()) {
      const string& x = 
        ::backend::OpDecl::GetOriginName(dx);
      CHECK(FindEdge(x) != NULL);
      Edge* dx_edge = FindEdge(dx, true);
      if (!dx_edge) {
        AddEdge(new Edge(dx, false, this));
        dx_edge = FindEdge(dx);
      }
      CHECK(dx_edge);
      dx_edge->AddSource(node);
      node->AddOutput(dx_edge);
    }

    for (auto& dy: op_def.input()) {
      const string& y = 
        ::backend::OpDecl::GetOriginName(dy);
      CHECK(FindEdge(y));
      CHECK(!FindEdge(dy));
      Edge* dy_edge = new Edge(dy, false, this);
      const_cast<Scope*>(this)->AddEdge(dy_edge);
      node->AddInput(dy_edge);
      dy_edge->AddDst(node);
    }
  }
}

void Scope::AddEdge(Edge* edge) {
  const string& name = edge->name();
  CHECK(edge_table_.find(name) ==
        edge_table_.end());
  edge_table_[name] = edge;
}

void Scope::PrintSymbolTable() {
  LOG(INFO) << "Printing Symbol Table";
  for (auto& one_pair : edge_table_) {
    LOG(INFO) << one_pair.first;
  }
}

const Scope* GetGlobalScope() {
  static Scope* s = new Scope(NULL, "global");
  return s;
}

} //namespace midend
