#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/session.h"

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Sym kernel1 = Sym::Variable(C_FLOAT, {20, 1, 5, 5 }, Sym::NormalRandom());
  Sym bias1   = Sym::Variable(C_FLOAT, {1, 20, 1, 1 }, Sym::Zeros());
  Sym kernel2 = Sym::Variable(C_FLOAT, {50, 20, 5, 5}, Sym::NormalRandom());
  Sym bias2   = Sym::Variable(C_FLOAT, {1, 50, 1, 1 }, Sym::Zeros());
  Sym fc1     = Sym::Variable(C_FLOAT, {500, 800    }, Sym::NormalRandom());
  Sym fc2     = Sym::Variable(C_FLOAT, {10, 500     }, Sym::NormalRandom());

  Sym image = Sym::MnistInput(100, "Image", "/users/shizhenx/projects/Cavs/apps/lenet-5/data");
  Sym label = Sym::MnistInput(100, "Label", "/users/shizhenx/projects/Cavs/apps/lenet-5/data");
  Sym y     = image.Conv(kernel1, bias1).Maxpooling(2, 2).Conv(kernel2, bias2).Maxpooling(2, 2).
              Flatten().FullyConnected(fc1).Relu().FullyConnected(fc2);
  Sym loss  = y.SoftmaxEntropyLogits(label);
  Sym train = loss.Optimizer();
  Sym correct_prediction = Sym::Equal(y.Argmax(1), label.Argmax(1)).Reduce_mean();
  Sym::DumpGraph();

  Session sess;
  for (int i = 0; i < 1; i++) {
    sess.Run({train, correct_prediction});
    loss.print();
  }
}
