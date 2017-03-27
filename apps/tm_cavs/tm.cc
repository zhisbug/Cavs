#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/session.h"

#include <stdlib.h>

DEFINE_int32(K, 100,  "num_of-topics");
DEFINE_int32(V, 1000, "vocab_size");
DEFINE_int32(D, 5000, "num_of_docs");
DEFINE_int32(num_epochs, 2, "num_of_epochs");
DEFINE_int32(inner_num_iters, 1, "num_of_inner_num_iters");
DEFINE_double(lr, 10, "learning_rate");
DEFINE_int32(mb_size, 5000, "size_of_minibatch");
DEFINE_string(file_docs, "/users/shizhenx/projects/Cavs/apps/tm_cavs/data/docs.dat", "doc_file");
DEFINE_int32(num_eval, 5000, "num_rand_smps_eval");

void load(void** doc_word) {
  *doc_word = malloc(FLAGS_D*FLAGS_V*sizeof(float));
  float *d_w = (float*)*doc_word;
  FILE *fp = fopen(FLAGS_file_docs.c_str(),"rb");
  if (!fp)
    LOG(FATAL) << "file[" << FLAGS_file_docs 
               << "] does not exists";
  for(int i = 0; i < FLAGS_D; i++) {
    fread(d_w+i*FLAGS_V, sizeof(float), FLAGS_V, fp);
  }
  fclose(fp);
}

int main() {
  void* doc_word_buf;
  load(&doc_word_buf);

  Sym doc_word = Sym::Placeholder(C_FLOAT, {FLAGS_mb_size, FLAGS_V});
  Sym doc_tpc  = Sym::Variable(C_FLOAT, {FLAGS_D, FLAGS_K}, Sym::UniformRandom(FLAGS_K));
  Sym tpc_word = Sym::Variable(C_FLOAT, {FLAGS_K, FLAGS_V}, Sym::UniformRandom(FLAGS_V));

  Sym loss = 0.5f/FLAGS_D*((doc_word-(Sym::MatMul(doc_tpc, tpc_word))).Square().Reduce_mean());
  Sym step1 = loss.Optimizer({doc_tpc}, FLAGS_lr, 20, "Simplex");
  Sym step2 = loss.Optimizer({tpc_word}, FLAGS_lr, 20, "Simplex");
  Sym::DumpGraph();

  Session sess;
  int iters = 200;
  for (int i = 0; i < iters; i++) {
    sess.Run({loss, step1, step2}, {{doc_word, doc_word_buf}});
    //sess.Run({loss, step1}, {{doc_word, doc_word_buf}});
    LOG(INFO) << "Iteration[" << i << "]:";
    loss.print();
  }

  return 0;
}
