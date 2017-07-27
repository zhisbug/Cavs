#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

DEFINE_int32 (batch,       20,       "batch");
DEFINE_int32 (input_size,  10000,    "input size");
DEFINE_int32 (timestep,    20,       "timestep");
DEFINE_int32 (hidden,      100,      "hidden size");
DEFINE_int32 (epoch,       1,        "epochs");
DEFINE_int32 (iters,       99999,    "iterations");
DEFINE_double(init_scale,  0.1f,     "init random scale of variables");
DEFINE_double(lr,          1.f,      "learning rate");
DEFINE_string(file_docs,
    "/users/shizhenx/projects/Cavs/apps/lstm/data/compressed.txt",
    "ptb_file");

void load(float** input_data, float** target, size_t* len) {
  vector<float> inputs;
  fstream file(FLAGS_file_docs);
  int id;
  int lines = 0; 
  while (file >> id && ++lines) {
    inputs.push_back(id);
  }
  cout <<  lines << endl;
  file.close();
  *len = inputs.size();
  cout << "Length:\t"<< *len << endl;
  *input_data = (float*)malloc(*len*sizeof(float));
  *target     = (float*)malloc(*len*sizeof(float));
  memcpy(*input_data, inputs.data(), *len*sizeof(float));
  memcpy(*target, inputs.data()+1, (*len-1)*sizeof(float));
}

class SeqModel : public GraphSupport {
 public:
  SeqModel(const Sym& graph_ph, const Sym& vertex_ph) :
    GraphSupport(graph_ph, vertex_ph) {
    int w_size  = 2*4*FLAGS_hidden*FLAGS_hidden;
    int b_size  = 4*FLAGS_hidden;
    embedding  = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    Sym LSTM_w = Sym::Variable(DT_FLOAT, {w_size},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    Sym LSTM_b = Sym::Variable(DT_FLOAT, {b_size},
                            //Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
                            Sym::Zeros());
    U  = LSTM_w.Slice(0, 4*FLAGS_hidden*FLAGS_hidden);
    W  = LSTM_w.Slice(4*FLAGS_hidden*FLAGS_hidden, 4*FLAGS_hidden*FLAGS_hidden);
    bi = LSTM_b.Slice(0, FLAGS_hidden);
    bf = LSTM_b.Slice(FLAGS_hidden, FLAGS_hidden);
    bu = LSTM_b.Slice(2*FLAGS_hidden, FLAGS_hidden);
    bo = LSTM_b.Slice(3*FLAGS_hidden, FLAGS_hidden);
  }

  void Node() override {
    Sym child = Gather(0, {2*FLAGS_hidden});
    Sym child_h, child_c;
    tie(child_h, child_c) = child.Split2();
    Sym x       = Pull(0, {1});
    x           = x.EmbeddingLookup(embedding.Mirror());

    Sym tmp = Sym::MatMul(x, U.Mirror().Reshape({FLAGS_hidden, 4*FLAGS_hidden}))
            + Sym::MatMul(child_h.Expand_dims(0), W.Mirror().Reshape({FLAGS_hidden, 4*FLAGS_hidden}));

    Sym i, f, u, o;
    tie(i, f, u, o) = tmp.Split4();

    i = (i+bi.Mirror()).Sigmoid();
    f = (f+bf.Mirror()).Sigmoid();
    u = (u+bu.Mirror()).Tanh();
    o = (o+bo.Mirror()).Sigmoid();

    Sym c = i * u + f*child_c;
    Sym h = o * Sym::Tanh(c.Mirror());

    Scatter(Sym::Concat({h.Mirror(), c.Mirror()}));
    Push(h.Mirror());
  }

 private:
  Sym U, W;
  Sym bu;
  Sym bo;
  Sym bi;
  Sym bf;
  Sym embedding;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  float *input_data, *label_data;
  size_t data_len;
  load(&input_data, &label_data, &data_len);
  vector<vector<float>> input_ph;
  vector<vector<float>> label_ph;
  vector<vector<int>> graph_ph;
  const int sample_len = data_len/FLAGS_batch;
  input_ph.resize(sample_len/FLAGS_timestep); 
  label_ph.resize(sample_len/FLAGS_timestep); 
  graph_ph.resize(sample_len/FLAGS_timestep); 
  for (int i = 0; i < sample_len/FLAGS_timestep; i++) {
    input_ph[i].resize(FLAGS_timestep*FLAGS_batch, 0);
    label_ph[i].resize(FLAGS_timestep*FLAGS_batch, 0);
    graph_ph[i].resize(FLAGS_timestep*FLAGS_batch, -1);
    for (int j = 0; j < FLAGS_batch; j++) {
      for (int k = 0; k < FLAGS_timestep; k++) {
        input_ph[i][j*FLAGS_timestep+k] = input_data[j*sample_len+i*FLAGS_timestep+k];
        label_ph[i][j*FLAGS_timestep+k] = label_data[j*sample_len+i*FLAGS_timestep+k]; 
        graph_ph[i][j*FLAGS_timestep+k] = (k+1 == FLAGS_timestep)? -1 : k+1;
      }
    }
  }

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, FLAGS_timestep}, "CPU");
  Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, FLAGS_timestep});
  Sym label    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, FLAGS_timestep});
  Sym weight   = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                               Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym bias     = Sym::Variable(DT_FLOAT, {1, FLAGS_input_size}, Sym::Zeros());

  SeqModel model(graph, word_idx);
  Sym graph_output = model.Output();
  Sym label_reshape = label.Reshape({-1, 1});
  label_reshape.ControlDependency(graph_output);
  Sym loss = graph_output.FullyConnected(weight, bias).SoftmaxEntropyLoss(label_reshape);
  Sym train      = loss.Optimizer({}, FLAGS_lr);
  Sym perplexity = loss.Reduce_mean();

  Session sess;
  LOG(INFO) << "here";
  int iterations = std::min(sample_len/FLAGS_timestep, FLAGS_iters);
  for (int i = 0; i < FLAGS_epoch; i++) {
    for (int j = 0; j < iterations; j++) {
      //sess.run({train}, {{input,input_ph[j%input_ph.size()].data()},
                         //{label,label_ph[j%label_ph.size()].data()}});
      sess.Run({train}, {{graph,    graph_ph[j%graph_ph.size()].data()},
                         {label,    label_ph[j%label_ph.size()].data()},
                         {word_idx, input_ph[j%input_ph.size()].data()}});
      LOG(INFO) << "Traing Epoch:\t" << i << "\tIteration:\t" << j;
    }
    //float sum = 0.f;
    //for (int j = 0; j < iterations; j++) {
      //sess.Run({perplexity}, {{graph,    graph_ph[j%graph_ph.size()].data()},
                              //{label,    label_ph[j%label_ph.size()].data()},
                              //{word_idx, input_ph[j%input_ph.size()].data()}});
      //float ppx = *(float*)(perplexity.eval());
      //LOG(INFO) << "Traing Epoch:\t" << i << "\tIteration:\t" << j
                //<< "\tPPX:\t" << exp(ppx);
      //sum += *(float*)(perplexity.eval());
    //}
    //LOG(INFO) << "Epoch[" << i << "]: loss = \t" << exp(sum/iterations);
    //float sum = 0.f;
    //for (int j = 0; j < iterations; j++) {
      //sess.Run({perplexity, train}, {{graph,    graph_ph[j%graph_ph.size()].data()},
                         //{label,    label_ph[j%label_ph.size()].data()},
                         //{word_idx, input_ph[j%input_ph.size()].data()}});
      //float ppx = *(float*)(perplexity.eval());
      //LOG(INFO) << "Traing Epoch:\t" << i << "\tIteration:\t" << j
                //<< "\tPPX:\t" << exp(ppx);
      //sum += *(float*)(perplexity.eval());
    //}
    //LOG(INFO) << "Epoch[" << i << "]: loss = \t" << exp(sum/iterations);
  }

  return 0;
}
