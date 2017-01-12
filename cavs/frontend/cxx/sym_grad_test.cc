#include "cavs/frontend/cxx/sym.h"

int main() {
  Sym A = Sym::Variable(C_FLOAT, {2, 3}); 
  Sym B = Sym::Placeholder(C_FLOAT, {2, 3});
  Sym C = A + B;
  Sym D = C.Optimizer();

  OpDef op_def;
  D.Finalize(&op_def);
  Sym::DumpGraph();
  LOG(INFO) << "\n" << op_def.DebugString();
  return 0;
}
