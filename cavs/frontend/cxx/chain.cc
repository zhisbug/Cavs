#include "cavs/frontend/cxx/chain.h"

const OpChainDef& Chain::Finalize() {
  op_chain_def_.reset(new OpChainDef);
  for (const sym* sym : syms_) {
    OpDef* op_def = op_chain_def->add_op();
    *op_def = sym.Finalize();
  }
  return *op_chain_def_;
}
