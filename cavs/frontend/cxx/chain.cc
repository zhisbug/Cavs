#include "cavs/frontend/cxx/chain.h"

void Chain::Finalize(cavs::OpChainDef* op_chain_def) const {
  for (auto* sym : syms_) {
    cavs::OpDef* op_def = op_chain_def->add_op();
    sym->Finalize(op_def);
  }
}
