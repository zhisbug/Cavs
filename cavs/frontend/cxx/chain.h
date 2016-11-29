#ifndef CHAIN_H_
#define CHAIN_H_

#include "cavs/frontend/cxx/sym.h"
#include "cavs/core/op_chain_def.pb.h"

class Chain {
 public:
  void push_back(const Sym* s) { syms_.push_back(s); }
  const OpChainDef& Finalize();

 private:
  vector<Sym*> syms_;
  unique_ptr<OpChainDef> op_chain_def_;
};

#endif

