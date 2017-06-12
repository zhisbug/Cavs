#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/util/macros_gpu.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

DEFINE_int32 (batch,       20,    "batch");
DEFINE_int32 (input_size,  10000, "input size");
DEFINE_int32 (timestep,    20,    "timestep");
DEFINE_int32 (hidden,      200,   "hidden size");
DEFINE_int32 (epoch,       10,    "epochs");
DEFINE_int32 (iters,       99999, "iterations");
DEFINE_double(init_scale,  0.1f,   "init random scale of variables");
DEFINE_double(lr,          1.f,   "learning rate");
DEFINE_string(file_docs,
    "/users/shizhenx/projects/Cavs/apps/lstm/data/compressed.txt",
    "ptb_file");

class SeqModel : public GraphSupport {
 public:
  SeqModel(const Sym& graph_ph, const Sym& vertex_ph) :
    GraphSupport(graph_ph, vertex_ph) {
    //It is the variable size required by cudnnRNN
    int var_size  = 2*4*(FLAGS_hidden*(FLAGS_hidden+1));
    embedding  = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    Sym LSTM_w = Sym::Variable(DT_FLOAT, {var_size},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    UW = LSTM_w.Slice(0, 2*4*FLAGS_hidden*FLAGS_hidden);
    bu = LSTM_w.Slice(2*4*FLAGS_hidden*FLAGS_hidden, FLAGS_hidden);
    bo = LSTM_w.Slice(2*4*FLAGS_hidden*FLAGS_hidden+FLAGS_hidden, FLAGS_hidden);
    bi = LSTM_w.Slice(2*4*FLAGS_hidden*FLAGS_hidden+2*FLAGS_hidden, FLAGS_hidden);
    bf = LSTM_w.Slice(2*4*FLAGS_hidden*FLAGS_hidden+3*FLAGS_hidden, FLAGS_hidden);
  }

  void Inode() override {
    Sym child_h = Gather(0, 0, {FLAGS_hidden, FLAGS_hidden});
    Sym child_c = Gather(0, FLAGS_hidden*FLAGS_hidden, {FLAGS_hidden, FLAGS_hidden});
    Sym x       = Pull(0, {FLAGS_input_size});
    x           = x.EmbeddingLookup(embedding);

    Sym xh = Sym::Concat({x, child_h});
    Sym tmp = Sym::MatMul(xh, UW);
    Sym u, i, o, f;
    tie(u, i, o, f) = tmp.Split4();

    i = (i+bi).Sigmoid();
    o = (o+bo).Sigmoid();
    u = (u+bu).Tanh();
    f = (f+bf).Sigmoid();

    Sym c = i * u + f*child_c;
    Sym h = o * Sym::Tanh(c);

    Scatter(Sym::Concat({h, c}));
    Push(h);
  }

  void Leaf() override {
    Sym x = Pull(0, {FLAGS_input_size});
    x = x.EmbeddingLookup(embedding);
    Sym h0 = Sym::Constant(DT_FLOAT, 0, {FLAGS_hidden*FLAGS_hidden});
    Sym xh = Sym::Concat({x, h0});
    Sym tmp = Sym::MatMul(xh, UW);
    Sym u, i, o, f;
    tie(u, i, o, f) = tmp.Split4();
    i = (i+bi).Sigmoid();
    o = (o+bo).Sigmoid();
    u = (u+bu).Tanh();
    Sym c = i * u;
    Sym h = o * Sym::Tanh(c);
    Push(h);
    Scatter(Sym::Concat({h, c}));
  }

 private:
  Sym UW;
  Sym bu;
  Sym bo;
  Sym bi;
  Sym bf;
  Sym embedding;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_log_dir =  "./";

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_timestep, FLAGS_batch});
  Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_timestep, FLAGS_batch});
  Sym label    = Sym::Placeholder(DT_FLOAT, {FLAGS_timestep, FLAGS_batch});
  Sym weight   = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                                Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym bias      = Sym::Variable(DT_FLOAT, {1, FLAGS_input_size}, Sym::Zeros());

  SeqModel model(graph, word_idx);
  Sym loss       = model.Output()
                        .Reshape({FLAGS_timestep*FLAGS_batch, FLAGS_hidden})
                        .FullyConnected(weight, bias)
                        .SoftmaxEntropyLoss(label.Reshape({FLAGS_timestep*FLAGS_batch,1}));
  Sym train      = loss.Optimizer({}, FLAGS_lr);
  Sym perplexity = loss.Reduce_mean();

  Session sess;
  int iterations = FLAGS_iters;
  for (int i = 0; i < FLAGS_epoch; i++) {
    for (int j = 0; j < iterations; j++) {
      //sess.Run({train}, {{input,input_ph[j%input_ph.size()].data()},
                         //{label,label_ph[j%label_ph.size()].data()}});
      LOG(INFO) << "Traing Epoch:\t" << i << "\tIteration:\t" << j;
    }
  }

  return 0;
}
