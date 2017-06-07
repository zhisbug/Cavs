#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/session.h"

#include <iostream>
#include <fstream>

DEFINE_int32 (K          , 1000, "num_of-topics"         );
DEFINE_int32 (V          , 61188, "vocab_size"            );
DEFINE_int32 (D          , 18774, "num_of_docs"           );
//DEFINE_int32 (D          , 4000, "num_of_docs"           );
DEFINE_int32 (epochs     , 200  , "num_of_epochs"         );
DEFINE_int32 (inner_iters, 20   , "num_of_inner_num_iters");
DEFINE_int32 (batch      , 400  , "size_of_minibatch"     );
DEFINE_double(lr         , 1    , "learning_rate"         );
DEFINE_string(file_docs,
    "/users/shizhenx/projects/Cavs/apps/topic_model_mf/data/20news_large.bin",
    "file_name");

int main(int argc, char* argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "DOC: " << FLAGS_D;
  Sym doc_word = Sym::Data(DT_FLOAT, {FLAGS_D, FLAGS_V}, FLAGS_batch,
                           Sym::BinaryReader(FLAGS_file_docs));
  Sym doc_tpc  = Sym::DDV(DT_FLOAT, {FLAGS_D, FLAGS_K}, FLAGS_batch,
                          Sym::UniformNormalizer(FLAGS_K));
  Sym tpc_word = Sym::Variable(DT_FLOAT, {FLAGS_K, FLAGS_V},
                               Sym::UniformNormalizer(FLAGS_V));

  Sym loss = 0.5f/FLAGS_batch*((doc_word-(Sym::MatMul(doc_tpc, tpc_word))).Square().Reduce_sum());
  Sym step1 = loss.Optimizer({doc_tpc}, FLAGS_lr, 0, FLAGS_inner_iters, "Simplex");
  Sym step2 = loss.Optimizer({tpc_word}, FLAGS_lr, 0, FLAGS_inner_iters, "Simplex");
  Sym::DumpGraph();

  Session sess;
  for (int i = 0; i < FLAGS_epochs; i++) {
    for (int j = 0; j < FLAGS_D/FLAGS_batch; j++) {
      sess.Run({loss, step1, step2});
      //sess.Run({step1, step2});
      LOG(INFO) << "Epoch[" << i << "]\t"
                << "Iteration[" << j << "]\n";
      loss.print();
    }
    float loss_sum = 0.f;
    for (int j = 0; j < FLAGS_D/FLAGS_batch; j++) {
      sess.Run({loss});
      loss_sum += *(float*)(loss.eval());
    }
    LOG(INFO) << "[Test] Epoch[" << i << "]:\t" << loss_sum/(FLAGS_D/FLAGS_batch);
  }

  return 0;
}
