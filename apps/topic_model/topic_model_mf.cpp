// topic_model_mf.cpp : Defines the entry point for the console application.
#include "tm_mf.h"
#include "paras.h"

int main(int argc, char* argv[]) {
	paras prs;
	const char *file_paras = "./data/paras.txt";
	load_paras(prs, file_paras);  
	print_paras(prs);

	tm_mf tm_mf_test(prs);
	tm_mf_test.learn();
	return 0;
}
