#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/session.h"

#include <iostream>
#include <fstream>

DEFINE_int32(K, 10000,  "num_of-topics");
DEFINE_int32(V, 10000, "vocab_size");
DEFINE_int32(D, 400, "num_of_docs");
//DEFINE_int32(num_epochs, 2, "num_of_epochs");
//DEFINE_int32(inner_num_iters, 1, "num_of_inner_num_iters");
DEFINE_double(lr, 10, "learning_rate");
DEFINE_int32(mb_size, 400, "size_of_minibatch");
DEFINE_string(file_docs,
    "/users/shizhenx/projects/Cavs/apps/tm_cavs/data/20news_large.txt",
    "doc_file");

void load_20news(void** doc_word) {
  *doc_word = malloc(FLAGS_D*FLAGS_V*sizeof(float));
  float *d_w = (float*)*doc_word;
  std::ifstream file(FLAGS_file_docs);
  const int DOC = 18774;
  const int WORD = 61188;
  CHECK(WORD >= FLAGS_V);
  if (file.is_open()) {
    for(int i = 0; i < FLAGS_D; i++) {
      float sum = 0.f;
      for (int j = 0; j < FLAGS_V; j++) {
        file >> d_w[j];
        sum += d_w[j];
        //if (i == 1)
          //LOG(INFO) << j << "\t" << d_w[j];
      }
      CHECK(sum != 0);
      for (int j = 0; j < FLAGS_V; j++) {
        d_w[j] /= sum;
        //if (i == 1)
          //LOG(INFO) << j << "\t" << d_w[j];
      }
      for (int j = FLAGS_V; j < WORD; j++) {
        int e; file >> e;
      }
      //LOG(INFO) << "line:" << i;
      d_w += FLAGS_V;
    }
  }else {
    LOG(FATAL) << "file[" << FLAGS_file_docs 
               << "] does not exists";
  }
  file.close();
}

int main() {
  void* doc_word_buf;
  load_20news(&doc_word_buf);
  LOG(INFO) << "Loading completes";

  Sym doc_word = Sym::Placeholder(C_FLOAT, {FLAGS_mb_size, FLAGS_V});
  Sym doc_tpc  = Sym::Variable(C_FLOAT, {FLAGS_D, FLAGS_K}, Sym::UniformRandom(FLAGS_K));
  Sym tpc_word = Sym::Variable(C_FLOAT, {FLAGS_K, FLAGS_V}, Sym::UniformRandom(FLAGS_V));

  Sym loss = 0.5f/FLAGS_D*((doc_word-(Sym::MatMul(doc_tpc, tpc_word))).Square().Reduce_mean());
  Sym step1 = loss.Optimizer({doc_tpc}, FLAGS_lr, 20, "Simplex");
  Sym step2 = loss.Optimizer({tpc_word}, FLAGS_lr, 20, "Simplex");
  Sym::DumpGraph();

  Session sess;
  int iters = 20;
  for (int i = 0; i < iters; i++) {
    sess.Run({loss, step1, step2}, {{doc_word, doc_word_buf}});
    LOG(INFO) << "Iteration[" << i << "]:";
    loss.print();
  }

  return 0;
}
