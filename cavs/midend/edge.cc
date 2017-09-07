#include "cavs/midend/edge.h"

using std::string;

namespace midend {

Edge::Edge(const string& name, Scope* s)
  : name_(name), located_(s), isDynamicEnabled_(false) {
  located_->AddEdge(this);
}

inline string Edge::scoped_name() const {
  return located_->scoped_name() + ":" + name();
}

void Edge::AddDst(Node* node) {
  dsts_.push_back(node); 
  if (node->scope() == scope())
    same_scoped_dsts_.push_back(node);
}

DataType Edge::dtype() const {
  CHECK(src_size() > 0 && src(0)->IsSingleNode());
  return dynamic_cast<SingleNode*>(src(0))->dtype();
}

void Edge::AddSource(Node* node) {
  //we loose the constraint to gradients.
  //In the lstm cases, one tensor should be 
  //transmitted to two or more operators, and therefore
  //the gradients come from two or more operators
  CHECK(isVariable() || isGradient() || srcs_.empty())
    << node->debug_info()
    << debug_info();
  srcs_.push_back(node);
  if (node->scope() == scope())
    same_scoped_srcs_.push_back(node);
}

void Edge::AddControlDependency(const Node* n) {
  CHECK(n->scope() == scope());
  control_dependency_on_me_.push_back(const_cast<Node*>(n));
}

string Edge::debug_info() const {
  string ret = "\nname:\t" + scoped_name() + 
               "\nshape:\t" + shape().DebugString() + 
               "scope:\t" + located_->name() +
               "\tsrcs_size:\t" + std::to_string(src_size()) + 
               "\ndsts_size:\t" + std::to_string(dst_size());
  for (int i = 0; i < src_size(); i++)
    ret += "\nsrc[" + std::to_string(i) + "]:\t" + src(i)->debug_info();
  for (int i = 0; i < dst_size(); i++)
    ret += "\ndst[" + std::to_string(i) + "]:\t" + dst(i)->debug_info();
  return ret;
}

} //namespace midend
