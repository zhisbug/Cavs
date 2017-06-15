#include "cavs/midend/scope.h"
#include "cavs/midend/graph_util.h"
#include "cavs/backend/op_decl.h"

#include <vector>

using std::string;
using std::vector;

namespace midend {

Scope::Scope(const Scope* father, const std::string& n)
    : father_(father) {
  if (father) {
    name_ = father->name() + ":" + n;
    CHECK(father->children_.find(name_) == father->children_.end());
    const_cast<Scope*>(father)->children_[name_] = const_cast<Scope*>(this);
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

Node* Scope::FindNode(const std::string& name) const {
  const Edge* edge = FindEdge(name);
  if (!edge) return NULL;
  CHECK(edge->isStateful() || edge->srcs_size() == 1)
    << edge->name() << edge->srcs_size();
  return edge->src(0);
}

//if node exists: return NULL;
//otherwise: return new allocated node;
Node* Scope::AddOp(const OpDef& op_def) {
  size_t hash_code = GetHash(op_def);
  if (hash_nodes_.find(hash_code) != hash_nodes_.end()) {
    LOG(WARNING) << "Duplicated node in current scope"
                 << op_def.DebugString();
    return NULL;
  }else {
    hash_nodes_.insert(hash_code);
  }

  SingleNode* node = new SingleNode(op_def, this);
  VLOG(V_DEBUG) << "Adding node \t" << node->DebugInfo()
                << "\tTo Scope " << name();
  VLOG(V_DEBUG) << "Adding its outputs...";
  for (auto& out : op_def.output()) {
    Edge* upper_out_edge = FindEdge(out, false);
    Edge* out_edge = FindEdge(out, true);
    //currently, only the source nodes can cross scopes
    if (node->isSourceOp()) {
      //CHECK(op_def.shape_size() > 0) << op_def.DebugString();
      VLOG_IF(V_DEBUG, op_def.shape_size() == 0 || op_def.shape(0).dim_size() == 0)
          << "No shape operator:\n"
          << op_def.DebugString();
      if (!upper_out_edge) {
        //CHECK(!father_);
        out_edge = new Edge(out, this);
        out_edge->AddSource(node);
        if (op_def.shape_size() > 0) {
          CHECK(op_def.shape_size() == 1);
          out_edge->SetShape(op_def.shape(0));
        }
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
  VLOG(V_DEBUG) << "Adding its inputs...";
                
  for (auto& input : op_def.input()) {
    const Edge* in_edge = FindEdge(input);
    CHECK(in_edge) << "name: " << input << "\n" << DebugInfo();
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

void Scope::GroupAllVariables(vector<string>* vars) const {
  for (Node* n : nodes_) {
    if (static_cast<SingleNode*>(n)->IsVariableOp()) {
      CHECK(n->outputs_size() == 1) << n->outputs_size();
      vars->push_back(n->output(0)->name());
    }
  }
}

Node* Scope::AddOptimizerOp(const OpDef& op_def) {
  return GraphUtil(this).AddOptimizerOp(op_def);
}

TensorShapeDef Scope::AddFunction(const FunctionDef& func_def) {
  CHECK(name_ == "global");
  CHECK(father_);
  return GraphUtil(this).AddFunction(func_def);
}

void Scope::DebugSymbolTable() {
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

Scope* global_scope() {
  static Scope* s = new Scope(NULL, "global");
  return s;
}

Scope* main_scope() {
  static Scope* s = new Scope(global_scope(), "main");
  return s;
}

} //namespace midend
