#include "cavs/frontend/cxx/session.h"
#include "cavs/util/logging.h"

#include <iostream>

using namespace std;

int main() {
  Sym A = Sym::Variable(C_FLOAT, {2, 3}, Sym::Ones()); 
  Sym B = Sym::Placeholder(C_FLOAT, {2, 3});
  Sym C = A * B;
  Sym D = C.Optimizer({A}, 5);

  MPISession sess;
  vector<float> A_data = {1, 2, 3, 4, 5, 6};
  vector<float> B_data = {6, 5, 4, 3, 2, 1};
  sess.Run(C, {{A, A_data.data()}, {B, B_data.data()}});
  C.print();
  return 0;
}
