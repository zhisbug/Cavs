#include "cavs/frontend/c_api.h"
#include "cavs/frontend/cxx/sym.h"
#include "cavs/util/logging.h"

int main() {
  Sym A = Sym::Placeholder(F_FLOAT, {2, 3}); 
  Sym B = Sym::Placeholder(F_FLOAT, {2, 3});
  Sym C = A + B;

  cavs::OpDef op_def;
  C.Finalize(&op_def);
  LOG(INFO) << "\n" << op_def.DebugString();
  return 0;
}

