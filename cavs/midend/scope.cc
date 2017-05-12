#include "scope.h"
#include "cavs/backend/op_decl.h"

#include <vector>

using std::string;
using std::vector;

namespace midend {

Scope::Scope(const Scope* s, const std::string& n)
    : father_(s) {
  if (s) {
    name_ = s->name() + ":" + n;
    CHECK(s->children_.find(name_) == s->children_.end());
    const_cast<Scope*>(s)->children_[name_] = const_cast<Scope*>(this);
  }else {
    name_ = n;
  }
}

Scope* Scope::FindChild(const string& n) const {
  if (children_.find(n) == children_.end())
    return NULL;
  else
    return const_cast<Scope*>(this)->children_[n]; 
}

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

//if node exists: return NULL;
//otherwise: return new allocated node;
Node* Scope::AddNode(const OpDef& op_def) {
  size_t hash_code = GetHash(op_def);
  if (hash_nodes_.find(hash_code) != hash_nodes_.end()) {
    LOG(WARNING) << "Duplicated node in current scope"
                 << op_def.DebugString();
    return NULL;
  }else {
    hash_nodes_.insert(hash_code);
  }

  SingleNode* node = new SingleNode(op_def, this);
  //LOG(INFO) << "Adding node \t" << node->DebugInfo()
            //<< "\nTo Scope " << name();
  for (auto& out : op_def.output()) {
    Edge* upper_out_edge = FindEdge(out, false);
    Edge* out_edge = FindEdge(out, true);
    //bool stateful = node->IsVariableOp();
    //currently, only the source nodes can cross scopes
    if (node->isSourceOp()) {
      //CHECK(op_def.shape_size() == 1) 
        //<< op_def.DebugString();
      LOG_IF(INFO, op_def.shape(0).dim_size() == 0)
          << op_def.DebugString();
      if (!upper_out_edge) {
        //CHECK(!father_);
        out_edge = new Edge(out, this);
        out_edge->AddSource(node);
        out_edge->SetShape(op_def.shape(0));
        node->AddOutput(out_edge);
      }else {
        CHECK(father_ && !out_edge);
        upper_out_edge->AddSource(node);
        node->AddOutput(upper_out_edge);
      }
    }else {
      if (!out_edge) {
        if (upper_out_edge){
          if (upper_out_edge->isStateful()) {
            out_edge = upper_out_edge;
          }else {
            out_edge = new Edge(out, this);
            out_edge->SetShape(upper_out_edge->shape());
          }
        }else {
          //LOG(INFO) << "No shape currently";
          out_edge = new Edge(out, this);
        }
      }
      out_edge->AddSource(node);
      node->AddOutput(out_edge);
      if (upper_out_edge && !upper_out_edge->isStateful()) {
        for (int i = 0; i < upper_out_edge->dsts_size(); i++) {
          if (upper_out_edge->dst(i)->scope() == this) {
            const_cast<Node*>(upper_out_edge->dst(i))
              ->replaceInput(i, out_edge);
          }
        }
      }
    }
  }
  //PrintSymbolTable();
  for (auto& input : op_def.input()) {
    const Edge* in_edge = FindEdge(input);
    CHECK(in_edge) << "name: " << input
      << DebugInfo();
    //LOG(INFO) << in_edge->name()
              //<< "???\t"
              //<< in_edge->scope()->name()
              //<< in_edge->shape().DebugString();
    node->AddInput(in_edge);
    const_cast<Edge*>(in_edge)->AddDst(node);
    if (!FindEdge(input, true) &&
        in_edges_.find(in_edge->name()) == in_edges_.end()) {
      in_edges_[in_edge->name()] = const_cast<Edge*>(in_edge);
    }
  }
  return node;
}

void Scope::AddNode(const Node* node) {
  CHECK(node->scope() == this);
  nodes_.push_back(const_cast<Node*>(node));
}

void Scope::AddEdge(const Edge* edge) {
  CHECK(edge->scope() == this);
  const string& name = edge->name();
  CHECK(edge_table_.find(name) ==
        edge_table_.end())
      << "Adding duplicated Edge: \"" << name << "\"";
  edge_table_[name] = const_cast<Edge*>(edge);
}

void Scope::PrintSymbolTable() {
  LOG(INFO) << "Printing Symbol Table\t" << name_;
  for (auto& one_pair : edge_table_) {
    LOG(INFO) << one_pair.first;
  }
}

string Scope::DebugInfo() {
  string ret = "\n============================\n";
  ret += "<<<<<<<<< In Scope " + name_ + " <<<<<<<<<\n";
  int i = 0;
  for (auto* node : nodes_) {
    ret += "The " + std::to_string(i++)
          + "th operators:\n";
    ret += node->op_def().DebugString();
    ret += "\n\n";
  }
  for (auto& child: children_) {
    ret += child.second->DebugInfo();
  }
  ret += "\n============================\n";
  return ret;
}

Scope* GetGlobalScope() {
  static Scope* s = new Scope(NULL, "global");
  return s;
}

} //namespace midend
