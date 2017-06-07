#include "cavs/frontend/cxx/sym.h"
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
DEFINE_int32 (lstm_layers, 2,     "stacked lstm layers");
DEFINE_int32 (epoch,       10,    "epochs");
DEFINE_int32 (iters,       99999, "iterations");
DEFINE_double(init_scale,  0.1f,   "init random scale of variables");
DEFINE_double(lr,          1.f,   "learning rate");
DEFINE_string(file_docs,
    "/users/shizhenx/projects/Cavs/apps/lstm/data/compressed.txt",
    "ptb_file");

void load(float** input_data, float** target, size_t* len) {
}

class TreeModel : Vertex {
 public:
  TreeModel() {
    //It is the variable size required by cudnnRNN
    int var_size  = 2*4*(FLAGS_hidden*(FLAGS_hidden+1));
    embedding  = Sym::Variable(C_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    Sym LSTM_w = Sym::Variable(C_FLOAT, {var_size},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    UW_uio = LSTM_w.Slice(0, 2*3*FLAGS_hidden*FLAGS_hidden);
    UW_f   = LSTM_w.Slice(2*3*FLAGS_hidden*FLAGS_hidden, FLAGS_hidden*FLAGS_hidden);
    bu     = LSTM_w.Slice(2*4*FLAGS_hidden*FLAGS_hidden, FLAGS_hidden);
    bo     = LSTM_w.Slice(2*4*FLAGS_hidden*FLAGS_hidden+FLAGS_hidden, FLAGS_hidden);
    bi     = LSTM_w.Slice(2*4*FLAGS_hidden*FLAGS_hidden+2*FLAGS_hidden, FLAGS_hidden);
    bf     = LSTM_w.Slice(2*4*FLAGS_hidden*FLAGS_hidden+3*FLAGS_hidden, FLAGS_hidden);
  }

  void Inode() override {
    Sym child_hl = InEdge(0).data(0); 
    Sym child_hr = InEdge(1).data(0); 
    Sym child_cl = InEdge(0).data(1);
    Sym child_cr = InEdge(1).data(1);
    Sym x = InData(0);

    Sym xh = Sym::Concat({x, child_hl+child_hr});
    Sym tmp = Sym::Matmul(xh, UW_uio);
    Sym u, i, o;
    tie(u, i, o) = tmp.split3(tmp, 1);
    Sym xhl = Sym::Concat({x, child_hl});
    Sym xhr = Sym::Concat({x, child_hr});
    Sym fl = Sym::Matmul(xhl, UW_f);
    Sym fr = Sym::Matmul(xhr, UW_f);

    i = (i+bi).Sigmoid();
    o = (o+bo).Sigmoid();
    u = (u+bu).Tanh();
    fl = (fl+bf).Sigmoid();
    fr = (fr+bf).Sigmoid();

    f = Sym::Concat({fl, fr});
    child_c = Sym::Concat({child_cl, child_cr});
    c = i * u + Sym::Reduce_sum(f*child_c, 0);
    Sym h = o * Sym::Tanh(c);

    OutData(0) = h;
    OutEdge(0).data(0) = h;
    OutEdge(0).data(1) = c;
  }

  void Leaf() override {
    Sym x = InData(0);
    x = x.EmbeddingLookup(embedding);
    Sym h0 = Sym::Constant(0, {FLAGS_hidden*FLAGS_hidden});
    Sym xh = Sym::Concat({x, h0});
    Sym tmp = Sym::Matmul(xh, UW_uio);
    Sym u, i, o;
    tie(u, i, o) = tmp.split3(tmp, 1);
    i = (i+bi).Sigmoid();
    o = (o+bo).Sigmoid();
    u = (u+bu).Tanh();
    c = i * u;
    Sym h = o * Sym::Tanh(c);
    OutData(0) = h;
    OutEdge(0).data(0) = h;
    OutEdge(0).data(1) = c;
  }

 private:
  Sym UW_uio;
  Sym UW_f;
  Sym bu;
  Sym bo;
  Sym bi;
  Sym bf;
  Sym embedding;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_log_dir =  "./";

  Sym input     = Sym::Placeholder(C_FLOAT, {FLAGS_timestep, FLAGS_batch});
  Sym label     = Sym::Placeholder(C_FLOAT, {FLAGS_timestep, FLAGS_batch});
  Sym weight    = Sym::Variable(C_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                                Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym bias      = Sym::Variable(C_FLOAT, {1, FLAGS_input_size}, Sym::Zeros());
  TreeModel model();
  Sym loss       = model.Output()
                        .Reshape({FLAGS_timestep*FLAGS_batch, FLAGS_hidden})
                        .FullyConnected(weight, bias)
                        .SoftmaxEntropyLoss(label.Reshape({FLAGS_timestep*FLAGS_batch,1}));
  Sym train      = loss.Optimizer({}, FLAGS_lr, 5);
  Sym perplexity = loss.Reduce_mean();

  Session sess;
  int iterations = std::min(sample_len/FLAGS_timestep, FLAGS_iters);
  for (int i = 0; i < FLAGS_epoch; i++) {
    for (int j = 0; j < iterations; j++) {
      sess.Run({train}, {{input,input_ph[j%input_ph.size()].data()},
                         {label,label_ph[j%label_ph.size()].data()}});
      LOG(INFO) << "Traing Epoch:\t" << i << "\tIteration:\t" << j;
    }
  }

  return 0;
}
