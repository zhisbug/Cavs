#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/session.h"

int main() {
  Sym kernel1 = Sym::Variable(C_FLOAT, {20, 1, 5, 5});
  Sym bias1 = Sym::Variable(C_FLOAT, {1, 20, 1, 1});
  Sym kernel2 = Sym::Variable(C_FLOAT, {50, 20, 5, 5});
  Sym bias2 = Sym::Variable(C_FLOAT, {1, 50, 1, 1});
  Sym fc1 = Sym::Variable(C_FLOAT, {500, 800});
  Sym fc2 = Sym::Variable(C_FLOAT, {10, 500});

  Sym image = Sym::MnistInput(100, "Image", "/users/shizhenx/projects/Cavs/apps/lenet-5/data");
  Sym label = Sym::MnistInput(100, "Label", "/users/shizhenx/projects/Cavs/apps/lenet-5/data");
  Sym loss = image.Conv(kernel1, bias1).Maxpooling(2, 2).Conv(kernel2, bias2).Maxpooling(2, 2).
              Flatten().FullyConnected(fc1).Relu().FullyConnected(fc2).SoftmaxEntropyLogits(label);
  Sym train = loss.Optimizer();
  Sym::DumpGraph();

  Session sess;
  sess.Run({train});
}
