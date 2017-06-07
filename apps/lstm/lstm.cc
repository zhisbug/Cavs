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
  vector<float> inputs;
  fstream file(FLAGS_file_docs);
  int id;
  int i = 0; 
  while (file >> id) {
    inputs.push_back(id);
    i++;
  }
  cout <<  i << endl;
  file.close();
  *len = inputs.size();
  cout << "Length:\t"<< *len << endl;
  *input_data = (float*)malloc(*len*sizeof(float));
  *target     = (float*)malloc(*len*sizeof(float));
  memcpy(*input_data, inputs.data(), *len*sizeof(float));
  memcpy(*target, inputs.data()+1, (*len-1)*sizeof(float));
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_log_dir =  "./";

  float *input_data, *label_data;
  size_t data_len;
  load(&input_data, &label_data, &data_len);
  vector<vector<float>> input_ph;
  vector<vector<float>> label_ph;
  const int sample_len = data_len/FLAGS_batch;
  cout << "sample_len = " << sample_len << endl;;
  input_ph.resize(sample_len/FLAGS_timestep); 
  label_ph.resize(sample_len/FLAGS_timestep); 
  for (int i = 0; i < sample_len/FLAGS_timestep; i++) {
    input_ph[i].resize(FLAGS_timestep*FLAGS_batch, 0);
    label_ph[i].resize(FLAGS_timestep*FLAGS_batch, 0);
    for (int j = 0; j < FLAGS_timestep; j++) {
      for (int k = 0; k < FLAGS_batch; k++) {
        input_ph[i][j*FLAGS_batch+k] = input_data[k*sample_len+i*FLAGS_timestep+j];
        label_ph[i][j*FLAGS_batch+k] = label_data[k*sample_len+i*FLAGS_timestep+j]; 
      }
    }
  }
  //for (int i = 0; i < FLAGS_batch; i++) {
    //cout << "[";
    //for (int j = 0; j < FLAGS_timestep; j++)
      //cout << input_data[i*sample_len+j] << "\t";
    //cout << "]\n";
  //}

  //int var_size = 4*(FLAGS_hidden*(FLAGS_input_size+(2*FLAGS_lstm_layers-1)*FLAGS_hidden))
                  //+ 4*2*FLAGS_lstm_layers*FLAGS_hidden;
  int var_size  = FLAGS_lstm_layers*2*4*(FLAGS_hidden*(FLAGS_hidden+1));
  Sym input     = Sym::Placeholder(DT_FLOAT, {FLAGS_timestep, FLAGS_batch});
  Sym label     = Sym::Placeholder(DT_FLOAT, {FLAGS_timestep, FLAGS_batch});
  Sym embedding = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                                Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym LSTM_w    = Sym::Variable(DT_FLOAT, {var_size},
                                Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym weight    = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                                Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym bias      = Sym::Variable(DT_FLOAT, {1, FLAGS_input_size}, Sym::Zeros());
  Sym loss      = input.EmbeddingLookup(embedding)
                       .LSTM(LSTM_w, FLAGS_lstm_layers, FLAGS_hidden)
                       .Reshape({FLAGS_timestep*FLAGS_batch, FLAGS_hidden})
                       .FullyConnected(weight, bias)
                       .SoftmaxEntropyLoss(label.Reshape({FLAGS_timestep*FLAGS_batch,1}));
  Sym train     = loss.Optimizer({}, FLAGS_lr, 5);
  Sym perplexity = loss.Reduce_mean();

  Session sess;
  int iterations = std::min(sample_len/FLAGS_timestep, FLAGS_iters);
  //int iterations = 20;
  for (int i = 0; i < FLAGS_epoch; i++) {
    for (int j = 0; j < iterations; j++) {
      sess.Run({train}, {{input,input_ph[j%input_ph.size()].data()},
                         {label,label_ph[j%label_ph.size()].data()}});
      LOG(INFO) << "Traing Epoch:\t" << i << "\tIteration:\t" << j;
    }
    float sum = 0.f;
    for (int j = 0; j < iterations; j++) {
      sess.Run({perplexity}, {{input,input_ph[j%input_ph.size()].data()},
                              {label,label_ph[j%label_ph.size()].data()}});
      float ppx = *(float*)(perplexity.eval());
      LOG(INFO) << "Traing Epoch:\t" << i << "\tIteration:\t" << j
                << "\tPPX:\t" << exp(ppx);
      sum += *(float*)(perplexity.eval());
    }
    LOG(INFO) << "Epoch[" << i << "]: loss = \t" << exp(sum/iterations);
  }

  free(input_data);
  free(label_data);

  return 0;
}
