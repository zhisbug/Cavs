#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

DEFINE_int32 (batch_size,  20,       "batch");
DEFINE_int32 (input_size,  21701,    "input size");
//DEFINE_int32 (timestep,    20,       "timestep");
DEFINE_int32 (embedding,   300,      "embedding size");
DEFINE_int32 (hidden,      150,      "hidden size");
DEFINE_int32 (epoch,       1,        "epochs");
DEFINE_int32 (iters,       99999,    "iterations");
DEFINE_double(init_scale,  0.1f,     "init random scale of variables");
DEFINE_double(lr,          1.f,      "learning rate");

int MAX_LEN = 56;
int MAX_DEPENDENCY = 111;
int NUM_SAMPLES = 8544;

DEFINE_string(input_file, "/users/shizhenx/projects/Cavs/apps/lstm/data/sst/train/sents_idx.txt", "input sentences");
DEFINE_string(label_file, "/users/shizhenx/projects/Cavs/apps/lstm/data/sst/train/labels.txt",    "label sentences");
DEFINE_string(graph_file, "/users/shizhenx/projects/Cavs/apps/lstm/data/sst/train/parents.txt",   "graph dependency");

class Reader {
 public:
  Reader(const string input, const string label, const string graph) :
      input_path(input), label_path(label), graph_path(graph),
      input_file(input), label_file(label), graph_file(graph) {
    //input_file = std::move(ifstream(input_path));
    //label_file = std::move(ifstream(label_path));
    //graph_file = std::move(ifstream(graph_path));
  }

  void next_batch( vector<int>* batch_graph, vector<float>* batch_input, vector<float>* batch_label) {
    std::fill(batch_input->begin(), batch_input->end(), 0);
    std::fill(batch_label->begin(), batch_label->end(), -1);
    std::fill(batch_graph->begin(), batch_graph->end(), -1);
    int i = 0; 
    int label_length = 0;
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
        int length;
        process_graph<int>(batch_graph->data() + i*MAX_DEPENDENCY, &length, graph_str);
        CHECK(MAX_DEPENDENCY >= length);

        process_data<float>(batch_input->data() + i*MAX_DEPENDENCY, &length, input_str);
        CHECK(MAX_LEN >= length);

        process_data<float>(batch_label->data() + label_length, &length, label_str);
        CHECK(MAX_DEPENDENCY >= length);
        label_length += length;
        i++;
      }
    }
  }

 private:
  template<typename T>
  void process_data(T* data, int* len, const string& str) {
    stringstream input_stream(str);
    int val, idx = 0;
    while (input_stream >> val) {
      data[idx] = val; 
      idx++;
    }
    *len = idx;
  }

  template<typename T>
  void process_graph(T* data, int* len, const string& str) {
    stringstream input_stream(str);
    int val, idx = 0;
    while (input_stream >> val) {
      data[idx] = val-1;
      idx++;
    }
    *len = idx;
    CHECK(data[idx-1] == -1);
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
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));//weight_5

    W = Sym::Variable(DT_FLOAT, {4 * FLAGS_embedding * FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));//weight_6
    U = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden * FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));//weight_7
    B = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden}, Sym::Zeros());//weight_8

    // prepare parameter symbols
    b_i = B.Slice(0, FLAGS_hidden);//slice_9
    b_f = B.Slice(FLAGS_hidden, FLAGS_hidden);//slice_10
    b_u = B.Slice(2 * FLAGS_hidden, FLAGS_hidden);//slice_11
    b_o = B.Slice(3 * FLAGS_hidden, FLAGS_hidden);//slice_12

    U_iou = U.Slice(0, 3 * FLAGS_hidden * FLAGS_hidden).Reshape({FLAGS_hidden, 3 * FLAGS_hidden});//slice_13
    U_f   = U.Slice(3 * FLAGS_hidden * FLAGS_hidden, FLAGS_hidden * FLAGS_hidden).Reshape({FLAGS_hidden, FLAGS_hidden});//slice_14
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

    // layout: i, o, u, f
    // start computation
    // xW is 1 x 4*FLAGS_hidden
    Sym xW = Sym::MatMul(x, W.Reshape({FLAGS_embedding, 4 * FLAGS_hidden}).Mirror()).Reshape({FLAGS_hidden * 4});
    Sym xW_i, xW_o, xW_u, xW_f;
    tie(xW_i, xW_o, xW_u, xW_f) = xW.Split4();
    
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
  Sym b_i;
  Sym b_f;
  Sym b_u;
  Sym b_o;
            
  Sym U_iou;
  Sym U_f;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  Reader sst_reader(FLAGS_input_file, FLAGS_label_file, FLAGS_graph_file);

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, MAX_DEPENDENCY}, "CPU");//placeholder_0
  //Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, MAX_LEN});
  Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, MAX_DEPENDENCY});//placeholder_1
  Sym label    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, MAX_DEPENDENCY});//placeholder_2

  Sym weight   = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                               Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));//weight_3
  Sym bias     = Sym::Variable(DT_FLOAT, {1, FLAGS_input_size}, Sym::Zeros());//weight_4
  TreeModel model(graph, word_idx);
  Sym graph_output = model.Output();
  Sym label_reshape = label.Reshape({-1, 1});
  label_reshape.ControlDependency(graph_output);
  Sym loss = graph_output.FullyConnected(weight, bias).SoftmaxEntropyLoss(label_reshape);
  //Sym train      = loss.Optimizer({}, FLAGS_lr);
  Sym perplexity = loss.Reduce_mean();
  Session sess;
  int iterations = NUM_SAMPLES / FLAGS_batch_size; 
  //vector<float> input_data(FLAGS_batch_size*MAX_LEN, -1);
  vector<float> input_data(FLAGS_batch_size*MAX_DEPENDENCY, -1);
  vector<float> label_data(FLAGS_batch_size*MAX_DEPENDENCY, -1);
  vector<int>   graph_data(FLAGS_batch_size*MAX_DEPENDENCY, -1);
  for (int i = 0; i < 26; i++)
    sst_reader.next_batch(&graph_data, &input_data, &label_data);
  for (int i = 0; i < FLAGS_epoch; i++) {
    float sum = 0.f;
    for (int j = 0; j < iterations; j++) {
      LOG(INFO) << "Traing Epoch:\t" << i << "\tIteration:\t" << j;
      sst_reader.next_batch(&graph_data, &input_data, &label_data);
      sess.Run({perplexity}, {{graph,    graph_data.data()},
                              {label,    label_data.data()},
                              {word_idx, input_data.data()}});
      float ppx = *(float*)(perplexity.eval());
      LOG(INFO) << "Traing Epoch:\t" << i << "\tIteration:\t" << j
                << "\tPPX:\t" << exp(ppx);
      sum += *(float*)(perplexity.eval());
    }
    LOG(INFO) << "Epoch[" << i << "]: loss = \t" << exp(sum/iterations);
  }

  return 0;
}
