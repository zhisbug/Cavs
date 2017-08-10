#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"
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

  Sym X = Sym::Placeholder(DT_FLOAT, {2, 2}); 
  Sym Y = Sym::Placeholder(DT_FLOAT, {2, 2});
  Sym Z = Sym::Placeholder(DT_FLOAT, {2, 2});
  Sym XX = X + Y;
  Sym YY = XX + Z;

  Session sess((int)OPT_FUSION);
  vector<float> A_data = {1, 2, 3, 4, 5, 6};
  vector<float> B_data = {6, 5, 4, 3, 2, 1};
  vector<float> C_data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

  vector<float> X_data = {1, 2, 3, 4};
  vector<float> Y_data = {6, 5, 4, 3};
  vector<float> Z_data = {0.1, 0.2, 0.3, 0.4};

  sess.Run({E, YY}, {{A, A_data.data()}, {B, B_data.data()}, {C, C_data.data()},
                     {X, X_data.data()}, {Y, Y_data.data()}, {Z, Z_data.data()}});
  E.print();
  YY.print();
  return 0;
}

