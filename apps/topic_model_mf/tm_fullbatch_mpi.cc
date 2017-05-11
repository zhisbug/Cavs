#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/session.h"

#include <stdlib.h>

DEFINE_int32(K, 100,  "num_of-topics");
DEFINE_int32(V, 1000, "vocab_size");
DEFINE_int32(D, 5000, "num_of_docs");
DEFINE_int32(iters, 200, "iterations");
DEFINE_double(lr, 10, "learning_rate");
DEFINE_int32 (inner_iters, 20, "num_of_inner_num_iters");
DEFINE_string(file_docs,
    "/users/shizhenx/projects/Cavs/apps/topic_model_mf/data/docs.dat",
    "doc_file");

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

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_log_dir =  "./";

  void* doc_word_buf;
  load(&doc_word_buf);

  Sym doc_word = Sym::Placeholder(C_FLOAT, {FLAGS_D, FLAGS_V});
  Sym doc_tpc  = Sym::Variable(C_FLOAT, {FLAGS_D, FLAGS_K}, Sym::UniformNormalizer(FLAGS_K));
  Sym tpc_word = Sym::Variable(C_FLOAT, {FLAGS_K, FLAGS_V}, Sym::UniformNormalizer(FLAGS_V));

  Sym loss = 0.5f/FLAGS_D*((doc_word-(Sym::MatMul(doc_tpc, tpc_word))).Square().Reduce_sum());
  Sym step1 = loss.Optimizer({doc_tpc}, FLAGS_lr, 0, FLAGS_inner_iters, "Simplex");
  Sym step2 = loss.Optimizer({tpc_word}, FLAGS_lr, 0, FLAGS_inner_iters, "Simplex");
  Sym::DumpGraph();

  MPISession sess;
  for (int i = 0; i < FLAGS_iters; i++) {
    sess.Run({loss, step1, step2}, {{doc_word, doc_word_buf}});
    LOG(INFO) << "Iteration[" << i << "]:";
    loss.print();
  }

  return 0;
}
