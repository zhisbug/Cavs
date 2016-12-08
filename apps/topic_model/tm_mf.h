#ifndef TM_MF_H_
#define TM_MF_H_

#include "paras.h"

class tm_mf {
 public:
  tm_mf(paras prs);
  ~tm_mf();
  void learn();
  
  //meta parameters
  int K; // number of topics
  int V; // vocabulary size
  int D; // number of docs
  //optimization parameters
  int num_epochs;
  int inner_num_iters;
  float lr; // learning rate
  int mb_size; // size of mini-batch
  int num_eval;

 private:
  void update_tpc_word();
  void infer();
  void infer_single_doc(int doc_id);
  void evaluate();
  void init_paras();
  void load_data(char * file_docs);
  
  float** tpc_word; // topic word matrix
  float** doc_tpc;  // doc topic matrix
  float** doc_word; // doc word matrix
  //context variable
  int* mb;
  int* eval_mb;
  //aux variables
  float* tmp_V;
  float* tmp_K;
  float** grad;
};

#endif
