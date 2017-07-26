#include "cavs/frontend/cxx/session.h"
#include "cavs/util/logging.h"

#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Sym A = Sym::Placeholder(DT_FLOAT, {2, 3}); 
  Sym B = Sym::Placeholder(DT_FLOAT, {2, 3});
  Sym C = Sym::Placeholder(DT_FLOAT, {2, 3});
  Sym D = A + B;
  Sym E = C + D;

  FusionSession sess;
  vector<float> A_data = {1, 2, 3, 4, 5, 6};
  vector<float> B_data = {6, 5, 4, 3, 2, 1};
  vector<float> C_data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  LOG(INFO) << "here";
  sess.Run({E}, {{A, A_data.data()},
                 {B, B_data.data()},
                 {C, C_data.data()}});
  LOG(INFO) << "here";
  E.print();
  return 0;
}

