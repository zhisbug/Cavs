#include "cavs/frontend/cxx/session.h"
#include "cavs/util/logging.h"

#include <iostream>

using namespace std;

int main() {
  Sym A = Sym::Variable(DT_FLOAT, {2, 3}, Sym::Ones()); 
  Sym B = Sym::Placeholder(DT_FLOAT, {2, 3});
  Sym C = A * B;
  Sym D = C.Optimizer({A}, 0.1f, 2);
  Sym::DumpGraph();

  MPISession sess;
  //Session sess;
  vector<float> B_data = {1, 2, 3, 4, 5, 6};
  sess.Run({A, D}, {{B, B_data.data()}});
  A.print();
  return 0;
}
