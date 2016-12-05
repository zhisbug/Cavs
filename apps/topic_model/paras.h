#ifndef PARAS_H_
#define PARAS_H_

#include <fstream>
#include <string>
#include <iostream>

struct paras
{
  //meta parameters
	int K; // number of topics
	int V; // vocabulary size
	int D; // number of docs

	//optimization parameters
	int num_epochs;
	int inner_num_iters;
	float lr; // learning rate
	int mb_size; // size of mini-batch
	char file_docs[256];
	int num_eval;// number of random samples for evaluation

};

void load_paras(paras & prs, const char * file_paras);
void print_paras(paras prs);

#endif

