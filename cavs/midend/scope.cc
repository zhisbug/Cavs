#include "cavs/midend/scope.h"
#include "cavs/midend/graph_util.h"
#include "cavs/backend/op_decl.h"

#include <vector>
#include <set>

using std::string;
using std::vector;
using std::set;

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

Scope* Scope::FindChildScope(const string& n) const {
  string scoped_name = name()+":"+n;
  if (children_.find(scoped_name) == children_.end())
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
  CHECK(edge->isStateful() || edge->src_size() == 1)
    << edge->name() << edge->src_size();
  return edge->src(0);
}

void Scope::AddGraphOpTransformation(OpDef* new_def, const OpDef& def) {
  VLOG(V_DEBUG) << "Original graph output info" << def.DebugString();
  set<string> inputs;
  for (auto& i : new_def->input())
    inputs.insert(i);
  for (auto&& func : {name()+":Inode", name()+":Leaf"}) {
    Scope* func_scope = this;
    //the function may defined in the current scope(current scope == main)
    //or the ancestor scope(current scope == optimizer)
    if (func_scope->children_.find(func) != func_scope->children_.end()) {
      func_scope = const_cast<Scope*>(func_scope->father_);
      if (func_scope->children_.find(func) != func_scope->children_.end()) {
        CHECK(func_scope->children_.find(func) != func_scope->children_.end())
             << func << "\n"
             << this->debug_info()
             << func_scope->debug_info();
      }
    }
    const Scope* c = func_scope->children_[func];
    CHECK_NOTNULL(c);
    for (auto& iter : c->in_edges_) {
      //if (iter.second->isStateful()) {
        //trainable_vars.insert(iter.second->name()); 
      //}
      inputs.insert(iter.second->name());
    }
  }

  new_def->clear_input(); 
  for (auto& s : inputs) {
    new_def->add_input(s); 
  }
  VLOG(V_DEBUG) << "new graph output info" << new_def->DebugString();
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

  SingleNode* node = NULL;
  OpDef new_def = op_def;
  if (op_def.name() == "GraphOutput") {
    AddGraphOpTransformation(&new_def, op_def);
  }
  node = new SingleNode(new_def, this);

  VLOG(V_DEBUG) << "Adding node \t" << node->debug_info()
                << "\tTo Scope " << name() << "\n"
                << new_def.DebugString();
  VLOG(V_DEBUG) << "Adding its outputs...";
  for (auto& out : new_def.output()) {
    Edge* upper_out_edge = FindEdge(out, false);
    Edge* out_edge = FindEdge(out, true);
    //currently, only the source nodes can cross scopes
    if (node->isSourceOp()) {
      VLOG_IF(V_DEBUG, new_def.shape_size() == 0 || new_def.shape(0).dim_size() == 0)
          << "No shape operator:\n"
          << new_def.DebugString();
      if (!upper_out_edge) {
        //CHECK(!father_);
        out_edge = new Edge(out, this);
        out_edge->AddSource(node);
        if (new_def.shape_size() > 0) {
          CHECK(new_def.shape_size() == 1);
          out_edge->SetShape(new_def.shape(0));
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
        for (int i = 0; i < upper_out_edge->dst_size(); i++) {
          if (upper_out_edge->dst(i)->scope() == this) {
            LOG(FATAL) << "It should not happen because we add each opeartor"
                       << "according to its dependency, "
                       << "both for user-defined operators and auto-diff operators";
            //const_cast<Node*>(upper_out_edge->dst(i))
              //->replaceInput(i, out_edge);
          }
        }
      }
    }
  }

  VLOG(V_DEBUG) << "Add its outputs done";
  VLOG(V_DEBUG) << "Adding its inputs...";
                
  for (auto& input : new_def.input()) {
    const Edge* in_edge = FindEdge(input);
    CHECK(in_edge) << "name: " << input << "\n" << debug_info();
    node->AddInput(in_edge);
    const_cast<Edge*>(in_edge)->AddDst(node);
    if (!FindEdge(input, true) &&
        in_edges_.find(in_edge->name()) == in_edges_.end()) {
      in_edges_[in_edge->name()] = const_cast<Edge*>(in_edge);
    }
  }

  VLOG(V_DEBUG) << "Add its inputs done";
  return node;
}

void Scope::AddNode(const Node* node) {
  CHECK(node->scope() == this);
  typological_sorted_nodes_.push_back(const_cast<Node*>(node));
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
  for (Node* n : typological_sorted_nodes_) {
    if (static_cast<SingleNode*>(n)->IsVariableOp()) {
      CHECK(n->output_size() == 1) << n->output_size();
      vars->push_back(n->output(0)->name());
    }
  }
}

Node* Scope::AddOptimizerOp(const OpDef& op_def) {
  return GraphUtil(this).AddOptimizerOp(op_def);
}

TensorShapeDef Scope::AddFunction(const FunctionDef& func_def) {
  CHECK(name_ == "main");
  CHECK(!father_);
  return GraphUtil(this).AddFunction(func_def);
}

void Scope::DebugSymbolTable() {
  LOG(INFO) << "Printing Symbol Table\t" << name_;
  for (auto& one_pair : edge_table_) {
    LOG(INFO) << one_pair.first;
  }
}

string Scope::debug_info() {
  string ret = "\n============================\n";
  ret += "<<<<<<<<< In Scope " + name_ + " <<<<<<<<<\n";
  int i = 0;
  for (auto* node : typological_sorted_nodes_) {
    ret += "The " + std::to_string(i++)
          + "th operators:\n";
    ret += node->op_def().DebugString();
    ret += "\n\n";
  }
  for (auto& child: children_) {
    ret += child.second->debug_info();
  }
  ret += "\n============================\n";
  return ret;
}

//Scope* global_scope() {
  //static Scope* s = new Scope(NULL, "global");
  //return s;
//}

Scope* main_scope() {
  static Scope* s = new Scope(NULL, "main");
  return s;
}

} //namespace midend
