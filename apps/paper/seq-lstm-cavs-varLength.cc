#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"
#include "cavs/util/timing.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

DEFINE_int32 (batch,       20,       "batch");
DEFINE_int32 (input_size,  10000,    "input size");
//DEFINE_int32 (timestep,    20,       "timestep");
DEFINE_int32 (hidden,      100,      "hidden size");
DEFINE_int32 (epoch,       1,        "epochs");
DEFINE_int32 (iters,       99,    "iterations");
DEFINE_double(init_scale,  0.1f,     "init random scale of variables");
DEFINE_double(lr,          1.f,      "learning rate");
DEFINE_string(file_docs,
    "/users/shizhenx/projects/Cavs/apps/lstm/data/train_to_idx.txt",
    "ptb_file");

const int MAX_LENGTH = 83;

void load(vector<vector<float>>* input_ph, vector<vector<float>>* label_ph, vector<vector<int>>* graph_ph) {
  ifstream file(FLAGS_file_docs);
  string s;
  CHECK(input_ph->empty());
  CHECK(label_ph->empty());
  CHECK(graph_ph->empty());
  bool end = false;
  while (!end) {
    vector<float> input_line(MAX_LENGTH*FLAGS_batch, -1);
    vector<float> label_line(MAX_LENGTH*FLAGS_batch, -1);
    vector<int>   graph_line(MAX_LENGTH*FLAGS_batch, -1);
    int outer_offset = 0;
    for (int sample_id = 0; sample_id < FLAGS_batch; sample_id++) {
      if (!getline(file, s)) {
        end = true;
        break; 
      }
      stringstream ss(s);
      float idx = 0;
      int current_offset = 0;
      while (ss >> idx) {
        input_line[current_offset + outer_offset] = idx;
        current_offset++;
      }
      memcpy(label_line.data()+outer_offset, input_line.data()+outer_offset+1, (current_offset-1)*sizeof(float));
      for (int i = 1; i < current_offset; i++) {
        graph_line[i-1+sample_id*MAX_LENGTH] = i;
      }
      outer_offset += current_offset;
    }

    if (end) {
      break;
    }else {
      input_ph->push_back(std::move(input_line));
      label_ph->push_back(std::move(label_line));
      graph_ph->push_back(std::move(graph_line));
    }
  }
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
    Sym LSTM_b = Sym::Variable(DT_FLOAT, {b_size}, Sym::Zeros());
    U  = LSTM_w.Slice(0, 4*FLAGS_hidden*FLAGS_hidden);//slice_8
    W  = LSTM_w.Slice(4*FLAGS_hidden*FLAGS_hidden, 4*FLAGS_hidden*FLAGS_hidden);//slice_9
    bi = LSTM_b.Slice(0, FLAGS_hidden);//slice_10
    bf = LSTM_b.Slice(FLAGS_hidden, FLAGS_hidden);//slice_11
    bu = LSTM_b.Slice(2*FLAGS_hidden, FLAGS_hidden);
    bo = LSTM_b.Slice(3*FLAGS_hidden, FLAGS_hidden);
  }

  void Node() override {
    Sym child = Gather(0, {2*FLAGS_hidden}); //gather_14
    Sym child_h/*slice_16*/, child_c/*slice_15*/;
    tie(child_h, child_c) = child.Split2();
    Sym x       = Pull(0, {1}); //pull_17
    x           = x.EmbeddingLookup(embedding/*.Mirror(<])[>mirror_18*/); //embeddinglookup_19

    Sym tmp = Sym::MatMul(x, U.Mirror()/*mirror_24*/.Reshape({FLAGS_hidden, 4*FLAGS_hidden})/*reshape_25*/)/*matmul_26*/
            + Sym::MatMul(child_h.Expand_dims(0)/*expand_dims_22*/,
                          W/*slice_9*/.Mirror()/*mirror_20*/.Reshape({FLAGS_hidden, 4*FLAGS_hidden})/*reshape_21*/)/*MatMul_23*/;//add_27

    Sym i, f, u, o;
    tie(i/*slice_31*/, f/*slice_30*/, u/*slice_29*/, o/*slice_28*/) = tmp.Split4();

    i = (i+bi.Mirror()/*mirror_32*/)/*add_33*/.Sigmoid();//sigmoid_34
    f = (f+bf.Mirror()/*mirror_35*/)/*add_36*/.Sigmoid();//sigmoid_37
    u = (u+bu.Mirror()/*mirror_38*/)/*add_39*/.Tanh();//tanh_40
    o = (o+bo.Mirror()/*mirror_41*/)/*add_42*/.Sigmoid();//sigmoid_43

    Sym c = i * u/*mul_45*/ + f*child_c/*mul_44*/;//add_46
    Sym h = o * Sym::Tanh(c.Mirror()/*mirror_47*/)/*tanh_48*/;//mul_49

    Scatter(Sym::Concat({h.Mirror()/*mirror_50*/, c.Mirror()/*mirror_51*/})/*concat_52*/);//scatter_53
    Push(h.Mirror()/*mirror_54*/);//push_55
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

  vector<vector<float>> input_ph;
  vector<vector<float>> label_ph;
  vector<vector<int>> graph_ph;
  load(&input_ph, &label_ph, &graph_ph);

  for (int i = 0; i < 3; i++) {
    for (int j  = 0; j < input_ph[i].size(); j++)
      cout << input_ph[i][j] << "\t";
    cout << "\n";
  }
  for (int i = 0; i < 3; i++) {
    for (int j  = 0; j < label_ph[i].size(); j++)
      cout << label_ph[i][j] << "\t";
    cout << "\n";
  }
  for (int i = 0; i < 3; i++) {
    for (int j  = 0; j < graph_ph[i].size(); j++)
      cout << graph_ph[i][j] << "\t";
    cout << "\n";
  }
  //LOG(FATAL) << "here";

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, MAX_LENGTH}, "CPU");
  Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, MAX_LENGTH});
  Sym label    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, MAX_LENGTH});
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

  //Session sess(OPT_BATCHING+OPT_FUSION+OPT_STREAMMING);
  //Session sess(OPT_BATCHING+OPT_STREAMMING);
  //Session sess(OPT_BATCHING+OPT_FUSION);
  //Session sess(OPT_FUSION+OPT_STREAMMING);
  //Session sess(OPT_BATCHING);
  //Session sess(OPT_FUSION);
  //Session sess(OPT_STREAMMING);
  Session sess;
  int iterations = FLAGS_iters;
  for (int i = 0; i < FLAGS_epoch; i++) {
    for (int j = 0; j < iterations; j++) {
      if (j % 10 == 0) {
        Timing::TimingBegin("Overall");
      }
      sess.Run({train}, {{graph,    graph_ph[j%graph_ph.size()].data()},
      //sess.Run({perplexity}, {{graph,    graph_ph[j%graph_ph.size()].data()},
                         {label,    label_ph[j%label_ph.size()].data()},
                         {word_idx, input_ph[j%input_ph.size()].data()}});
      if (j % 10 == 9) {
        Timing::TimingEnd("Overall");
        LOG(INFO) << "Iteration [" << j-9 << "-" << j << "]: Overall(ms): " << Timing::TimeInMs("Overall");
        LOG(INFO) << "Iteration [" << j-9 << "-" << j << "]: GraphParsing(ms): " << Timing::TimeInMs("GraphParsing");
        LOG(INFO) << "Iteration [" << j-9 << "-" << j << "]: RNNForward(ms): " << Timing::TimeInMs("RNNForward");
        LOG(INFO) << "Iteration [" << j-9 << "-" << j << "]: RNNBackward(ms): " << Timing::TimeInMs("RNNBackward");
        Timing::Reset("Overall");
        Timing::Reset("GraphParsing");
        Timing::Reset("RNNForward");
        Timing::Reset("RNNBackward");
      }
    }
  }

  return 0;
}
