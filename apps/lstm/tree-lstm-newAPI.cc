#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

DEFINE_int32 (batch_size,       20,       "batch");
DEFINE_int32 (input_size,  21701,    "input size");
DEFINE_int32 (timestep,    20,       "timestep");
DEFINE_int32 (embedding,   200,      "embedding size");
DEFINE_int32 (hidden,      100,      "hidden size");
DEFINE_int32 (epoch,       1,        "epochs");
DEFINE_int32 (iters,       99999,    "iterations");
DEFINE_double(init_scale,  0.1f,     "init random scale of variables");
DEFINE_double(lr,          1.f,      "learning rate");

int MAX_LEN = 56;
int MAX_DEPENDENCY = 111;
int NUM_SAMPLES = 8544;

DEFINE_string(input_file, "/users/hzhang2/projects/Cavs/apps/lstm/sst/train/sents_idx.txt", "input sentences");
DEFINE_string(label_file, "/users/hzhang2/projects/Cavs/apps/lstm/sst/train/labels.txt", "label sentences");
DEFINE_string(graph_file, "/users/hzhang2/projects/Cavs/apps/lstm/sst/train/parents.txt", "graph dependency");

class Reader {
 public:
  Reader(const string input, const string label, const string graph) :
      input_path(input), label_path(label), graph_path(graph) {
    input_file = ifstream(input_path);
    label_file = ifstream(label_path);
    graph_file = ifstream(graph_path);
  }

  void next_batch(vector<vector<int> >& batch_input, vector<vector<int> >& batch_label, 
      vector<vector<int> >& batch_graph) {
    batch_input.clear();
    batch_label.clear();
    batch_graph.clear();
    int i = 0; 
    while (i < FLAGS_batch_size) {
      if (input_file.eof()) { // which mean it reaches the end of the file
        input_file.clear();
        input_file.seekg(0, ios::beg);
        label_file.clear();
        label_file.seekg(0, ios::beg);
        graph_file.clear();
        graph_file.seekg(0, ios::beg);
      }

      string input_str, label_str, graph_str;
      getline(input_file, input_str);
      if (input_str.length() > 0) {
        getline(label_file, label_str);
        getline(graph_file, graph_str);
        batch_input.push_back(process_line(input_str, MAX_LEN));
        batch_label.push_back(process_line(label_str, MAX_DEPENDENCY));
        batch_graph.push_back(process_line(graph_str, MAX_DEPENDENCY)); 
        i++;
      }
    }
  }

 private:
  vector<int> process_line(const string str, const int max_len) {
    stringstream input_stream(str);
    vector<int> ret;
    ret.resize(max_len);
    for (int i = 0; i < max_len; ++i)
      ret[i] = -1;
    int val, idx = 0;
    while (input_stream >> val) {
      ret[idx] = val; 
      idx++;
    }
    return ret;
  }

  string input_path;
  string label_path;
  string graph_path;

  ifstream input_file;
  ifstream label_file;
  ifstream graph_file;
};

class TreeModel : public GraphSupport {
 public:
  TreeModel(const Sym& graph_ph, const Sym& vertex_ph) : 
    GraphSupport(graph_ph, vertex_ph) {
    embedding = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_embedding},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));

    W = Sym::Variable(DT_FLOAT, {4 * FLAGS_embedding * FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    U = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden * FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    B = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden}, Sym::Zeros());
  }

  void Node() override {
    // this 4 lines of code (interface) is a bit counter-intuitive that needs a revision
    // Gather from child nodes
    Sym left = Gather(0, {2 * FLAGS_hidden});
    Sym right = Gather(1, {2 * FLAGS_hidden});
    Sym h_l, c_l, h_r, c_r;
    tie(h_l, c_l) = left.Split2();
    tie(h_r, c_r) = right.Split2();
    Sym h_lr = h_l + h_r;
    
    // Pull the input word
    Sym x = Pull(0, {1});
    x = x.EmbeddingLookup(embedding.Mirror());

    // prepare parameter symbols
    Sym b_i = B.Slice(0, FLAGS_hidden);
    Sym b_f = B.Slice(FLAGS_hidden, FLAGS_hidden);
    Sym b_u = B.Slice(2 * FLAGS_hidden, FLAGS_hidden);
    Sym b_o = B.Slice(3 * FLAGS_hidden, FLAGS_hidden);

    // layout: i, o, u, f
    // start computation
    // xW is 1 x 4*FLAGS_hidden
    Sym xW = Sym::MatMul(x, W.Reshape({FLAGS_embedding, 4 * FLAGS_hidden}).Mirror()).Reshape({FLAGS_hidden * 4});
    Sym xW_i, xW_o, xW_u, xW_f;
    tie(xW_i, xW_o, xW_u, xW_f) = xW.Split4();

    Sym U_iou = U.Slice(0, 3 * FLAGS_hidden * FLAGS_hidden)
        .Reshape({FLAGS_hidden, 3 * FLAGS_hidden});
    Sym U_f = U.Slice(3 * FLAGS_hidden * FLAGS_hidden, FLAGS_hidden * FLAGS_hidden)
        .Reshape({FLAGS_hidden, FLAGS_hidden});
    
    // hU_iou is 1 x 3*FLAGS_hidden
    Sym hU_iou = Sym::MatMul(h_lr.Reshape({1, FLAGS_hidden}), U_iou.Mirror()).Reshape({FLAGS_hidden * 3});
    Sym hU_i, hU_o, hU_u;
    tie(hU_i, hU_o, hU_u) = hU_iou.Split3();

    // forget gate for every child
    Sym hU_fl = Sym::MatMul(h_l.Reshape({1, FLAGS_hidden}), U_f.Mirror()).Reshape({FLAGS_hidden});
    Sym hU_fr = Sym::MatMul(h_r.Reshape({1, FLAGS_hidden}), U_f.Mirror()).Reshape({FLAGS_hidden});

    // Derive i, f_l, f_r, o, u
    Sym i = (xW_i + hU_i + b_i.Mirror()).Sigmoid();
    Sym o = (xW_o + hU_o + b_o.Mirror()).Sigmoid();
    Sym u = (xW_u + hU_u + b_u.Mirror()).Tanh();

    Sym f_l = (xW_f + hU_fl + b_f.Mirror()).Sigmoid();
    Sym f_r = (xW_f + hU_fr + b_f.Mirror()).Sigmoid();

    Sym c = i * u + f_l * c_l + f_r * c_r;
    Sym h = o * Sym::Tanh(c.Mirror());

    Scatter(Sym::Concat({h.Mirror(), c.Mirror()}));
    Push(h.Mirror());
  }

 private:
  Sym W, U, B;
  Sym embedding;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  Reader sst_reader = Reader(FLAGS_input_file, FLAGS_label_file, 
      FLAGS_graph_file);
  vector<vector<int> > test_input, test_label, test_graph;

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, MAX_DEPENDENCY}, "CPU");
  Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, MAX_LEN});
  Sym label    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, MAX_DEPENDENCY});

  Sym weight   = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                               Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym bias     = Sym::Variable(DT_FLOAT, {1, FLAGS_input_size}, Sym::Zeros());
  TreeModel model(graph, word_idx);
  Sym graph_output = model.Output();
  Sym label_reshape = label.Reshape({-1, 1});
  label_reshape.ControlDependency(graph_output);
  Sym loss = graph_output.FullyConnected(weight, bias).SoftmaxEntropyLoss(label_reshape);
  LOG(INFO)  << "123123"  << endl;
  Sym train      = loss.Optimizer({}, FLAGS_lr);
  LOG(INFO)  << "123123"  << endl;
  Sym perplexity = loss.Reduce_mean();
  Session sess;
  int iterations = NUM_SAMPLES / FLAGS_batch_size; 
  vector<vector<int> > batch_input, batch_label, batch_graph;
  for (int i = 0; i < FLAGS_epoch; i++) {
    for (int j = 0; j < iterations; j++) {
      sst_reader.next_batch(batch_input, batch_label, batch_graph);
      sess.Run({train}, {{graph,    batch_graph.data()},
                         {label,    batch_label.data()},
                         {word_idx, batch_input.data()}});
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
