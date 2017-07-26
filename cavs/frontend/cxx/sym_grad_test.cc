#include "cavs/frontend/cxx/sym.h"

int main() {
  Sym A = Sym::Variable(DT_FLOAT, {2, 3}); 
  Sym B = Sym::Placeholder(DT_FLOAT, {2, 3});
  Sym C = A + B;
  Sym D = C.Optimizer();

  LOG(INFO) << "\n" << D.def().DebugString();
  return 0;
}
