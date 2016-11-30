#include "cavs/frontend/c_api.h"
#include "cavs/frontend/cxx/chain.h"
#include "cavs/util/logging.h"

int main() {
  Sym A = Sym::Placeholder(FLOAT, {2, 3}); 
  Sym B = Sym::Placeholder(FLOAT, {2, 3});
  Sym C = A + B;

  cavs::OpChainDef op_chain_def;
  Chain::Default()->Finalize(&op_chain_def);
  LOG(INFO) << "\n" << op_chain_def.DebugString();
  return 0;
}
