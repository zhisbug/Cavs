#include "cavs/frontend/cxx/sym.h"

int main() {
  Sym A = Sym::Placeholder(DT_FLOAT, {2, 3}); 
  Sym B = Sym::Placeholder(DT_FLOAT, {2, 3});
  Sym C = A + B;

  LOG(INFO) << "\n" << C.def().DebugString();
  return 0;
}
