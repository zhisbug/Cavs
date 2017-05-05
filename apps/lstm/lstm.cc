#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/session.h"

#include <iostream>
#include <fstream>

using namespace std;

DEFINE_int32 (batch,       32,    "batch");
DEFINE_int32 (input_size,  10000, "input size");
DEFINE_int32 (timestep,    20,    "timestep");
DEFINE_int32 (hidden,      200,   "hidden size");
DEFINE_int32 (lstm_layers, 3,     "stacked lstm layers");
DEFINE_int32 (iters,       200,   "iterations");
DEFINE_double(lr,          1.f,   "learning rate");
DEFINE_string(file_docs,
    "/users/shizhenx/projects/Cavs/apps/lstm/data/compressed.txt",
    "ptb_file");

void load(float** input_data, float** target, size_t* len) {
  vector<float> inputs;
  ifstream file;
  file.open(FLAGS_file_docs);
  CHECK(file.is_open());
  while (!file.eof()) {
    float id;
    file >> id;
    inputs.push_back(id);
    //cout << id << "\t";
  }
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
  for (int i = 0; i < sample_len/FLAGS_timestep; i++) {
    input_ph.resize(i+1); 
    label_ph.resize(i+1); 
    for (int j = 0; j < FLAGS_timestep; j++) {
      for (int k = 0; k < FLAGS_batch; k++) {
        for (int l = 0; l < FLAGS_input_size; l++) {
          input_ph[i].push_back(input_data[k*sample_len+i*FLAGS_timestep+j] == l? l : 0); 
        }
        label_ph[i].push_back(label_data[k*sample_len+i*FLAGS_timestep+j]); 
      } 
    }
  }

  Sym input = Sym::Placeholder(C_FLOAT, {FLAGS_timestep, FLAGS_batch, FLAGS_input_size});
  Sym label = Sym::Placeholder(C_FLOAT, {FLAGS_timestep, FLAGS_batch});
  Sym var   = Sym::Variable(C_FLOAT, {4*(FLAGS_hidden*(FLAGS_input_size+(2*FLAGS_lstm_layers-1)*FLAGS_hidden))});
  Sym loss  = input.LSTM(var, FLAGS_lstm_layers, FLAGS_hidden)
              .Reshape({FLAGS_timestep*FLAGS_batch, FLAGS_hidden}).SoftmaxEntropyLogits(label);
  Sym train = loss.Optimizer({}, FLAGS_lr);

  Session sess;
  for (int i = 0; i < FLAGS_iters; i++)
    sess.Run({loss, train}, {{input,input_ph[i%input_ph.size()].data()},
                             {label,label_ph[i%label_ph.size()].data()}});

  return 0;
}
