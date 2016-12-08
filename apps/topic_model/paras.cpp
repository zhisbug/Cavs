#include "paras.h"

#include <fstream>
#include <string>
#include <iostream>
#include <cassert>

using namespace std;

void load_paras(paras & prs, const char *file_paras) {
  ifstream infile;
  infile.open(file_paras);
  assert(infile.is_open());
  string tmp;
  infile >> tmp >> prs.K;// number of topics
  infile >> tmp >> prs.V;// vocabulary size
  infile >> tmp >> prs.D;// number of docs
  infile >> tmp >> prs.num_epochs;
  infile >> tmp >> prs.inner_num_iters;
  infile >> tmp >> prs.lr;
  infile >> tmp >> prs.mb_size;
  infile >> tmp >> prs.file_docs;
  infile >> tmp >> prs.num_eval;
  infile.close();
}

void print_paras(paras prs) {
  cout << "num of topics:"                  << prs.K << endl;
  cout << "vocab size:"                     << prs.V << endl;
  cout << "num of docs:"                    << prs.D<< endl;
  cout << "num of epochs:"                  << prs.num_epochs << endl;
  cout << "num of inner_num_iters:"         << prs.inner_num_iters << endl;
  cout << "learning rate:"                  << prs.lr << endl;
  cout << "size of minibatch:"              << prs.mb_size << endl;
  cout << "doc file:"                       << prs.file_docs << endl;
  cout << "num random smps for evaluation:" << prs.num_eval << endl;
}
