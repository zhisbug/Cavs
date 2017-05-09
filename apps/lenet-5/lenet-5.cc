#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/session.h"

DEFINE_int32 (iterations, 1     , "num_of_iterations");
DEFINE_int32 (batch     , 100   , "size_of_minibatch");
DEFINE_double(lr        , 0.05, "learning_rate"    );

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Sym kernel1  = Sym::Variable(C_FLOAT, {20, 1, 5, 5 }, Sym::Xavier());
  Sym bias1    = Sym::Variable(C_FLOAT, {1, 20, 1, 1 }, Sym::Zeros());
  Sym kernel2  = Sym::Variable(C_FLOAT, {50, 20, 5, 5}, Sym::Xavier());
  Sym bias2    = Sym::Variable(C_FLOAT, {1, 50, 1, 1 }, Sym::Zeros());
  Sym fc1      = Sym::Variable(C_FLOAT, {500, 800    }, Sym::Xavier());
  Sym bias_fc1 = Sym::Variable(C_FLOAT, {1, 500      }, Sym::Zeros());
  Sym fc2      = Sym::Variable(C_FLOAT, {10, 500     }, Sym::Xavier());
  Sym bias_fc2 = Sym::Variable(C_FLOAT, {1, 10       }, Sym::Zeros());

  Sym image = Sym::MnistInput(FLAGS_batch, "Image", "/users/shizhenx/projects/Cavs/apps/lenet-5/data");
  Sym label = Sym::MnistInput(FLAGS_batch, "Label", "/users/shizhenx/projects/Cavs/apps/lenet-5/data");
  Sym y     = image.Conv(kernel1, bias1).Maxpooling(2, 2).Conv(kernel2, bias2).Maxpooling(2, 2)
              .Flatten().FullyConnected(fc1, bias_fc1).Relu()
              .FullyConnected(fc2, bias_fc2).SoftmaxEntropyLogits(label);
  Sym train = y.Optimizer({}, FLAGS_lr);
  Sym correct_prediction = Sym::Equal(y.Argmax(1), label).Reduce_mean();
  Sym::DumpGraph();

  Session sess;
  for (int i = 0; i < FLAGS_iterations; i++) {
    sess.Run({train, correct_prediction});
    //sess.Run({train});
    LOG(INFO) << "Iteration[ " << i << "]: "
              << *(float*)correct_prediction.eval();
    //correct_prediction.print();
  }
}
