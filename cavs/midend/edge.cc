#include "cavs/midend/edge.h"

using std::string;

namespace midend {

Edge::Edge(const string& name, Scope* s)
  : name_(name), located_(s) {
  located_->AddEdge(this);
}

inline string Edge::scoped_name() const {
  return located_->name() + ":" + name();
}

void Edge::AddDst(Node* node) {
  dsts_.push_back(node); 
  if (node->scope() == scope())
    same_scoped_dsts_.push_back(node);
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
