#include "scope.h"

using std::string;

namespace midend {

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
  const string& name = node->name();
  Node* node = new Node(op_def);
  nodes_.push_back(node);
  node->set_id(nodes_.size());
  for (auto& out : op_def.output()) {
    Edge* out_edge = FindEdge(out);
    if (!out_edge) {
      bool stateful = (op_def.name() == "Variable");
      out_edge = new Edge(out, stateful, s);
      const_cast<Scope*>(s)->AddEdge(out_edge);
      out_edge->AddDst(sink_);
      //out_edge = FindEdge(out);
    }
    CHECK(out_edge);
    out_edge->AddSource(node);
    node->AddOutput(out_edge);
  }
  for (auto& input : op_def.input()) {
    const Edge* in_edge = FindEdge(input);
    CHECK(in_edge);
    node->AddInput(in_edge);
    const_cast<Edge*>(in_edge)->AddDst(node);
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
      CHECK(s->FindEdge(x) != NULL);
      Edge* dx_edge = s->FindEdge(dx);
      if (!dx_edge) {
        const_cast<Scope*>(s)->AddEdge(new Edge(dx, false, s));
        dx_edge = s->FindEdge(dx);
      }
      CHECK(dx_edge);
      dx_edge->AddSource(node);
      node->AddOutput(dx_edge);
    }

    for (auto& dy: op_def.input()) {
      const string& y = 
        ::backend::OpDecl::GetOriginName(dy);
      CHECK(s->FindEdge(y));
      CHECK(!(s->FindEdge(dy)));
      Edge* dy_edge = new Edge(dy, false, s);
      const_cast<Scope*>(s)->AddEdge(dy_edge);
      node->AddInput(dy_edge);
      dy_edge->AddDst(node);
    }
  }
  //grad_nodes_.push_back(std::move(grad_vec));
}

void Scope::AddEdge(Edge* edge) {
  const string& name = edge->name();
  CHECK(edge_table_.find(name) ==
        edge_table_.end());
  edge_table_[name] = edge;
}

const Scope* GetGlobalScope() {
  static Scope* s = new Scope(NULL, "global");
  return s;
}

} //namespace midend
