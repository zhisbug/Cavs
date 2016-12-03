#include "cavs/frontend/cxx/session.h"
#include "cavs/util/logging.h"

int main() {
  Sym A = Sym::Placeholder(F_FLOAT, {2, 3}); 
  Sym B = Sym::Placeholder(F_FLOAT, {2, 3});
  Sym C = A + B;

  cavs::OpChainDef op_chain_def;
  Chain::Default()->Finalize(&op_chain_def);

  Session sess;
  sess.Run({C});
  return 0;
}

