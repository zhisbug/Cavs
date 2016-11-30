#ifndef CAVS_FRONTEND_CXX_CHAIN_H_
#define CAVS_FRONTEND_CXX_CHAIN_H_

#include "cavs/midend/op_chain_def.pb.h"
#include "cavs/frontend/cxx/sym.h"

#include <vector>
#include <memory>
using std::vector;
using std::unique_ptr;

class Sym;
class Chain {
 public:
  void push_back(const Sym* s) { syms_.push_back(s); }
  void Finalize(cavs::OpChainDef* op_chain_def) const;

 private:
  vector<const Sym*> syms_;
};

#endif

