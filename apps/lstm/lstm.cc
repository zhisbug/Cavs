#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/session.h"

#include <iostream>
#include <fstream>

using namespace std;

DEFINE_double(lr, 10, "learning_rate");
DEFINE_int32 (iters, 200, "iterations");
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

  float *input_data, *target;
  size_t data_len;
  load(&input_data, &target, &data_len);

  return 0;
}
