#include "tm_mf.h"
#include "utils.h"

#include <iostream>

#define MIN_BETA 1e-30

void tm_mf::evaluate() {
	smp_mb(eval_mb, num_eval, D);
	float obj_sum=0;
	for(int i=0;i<num_eval;i++) {
		int doc_id=eval_mb[i];
		vec_mat_mul(tmp_V, doc_tpc[doc_id], tpc_word, K, V);
		vec_vec_add(tmp_V, doc_word[doc_id],-1, V);
		float los=vec_sq(tmp_V, V)/2;
		obj_sum+=los;
	}
	std::cout<<"objective value is "<<obj_sum<<std::endl;
}

void tm_mf::infer() {
	for(int i=0;i<mb_size;i++) {
		int doc_id=mb[i];
		infer_single_doc(doc_id);
	}
}

void tm_mf::infer_single_doc(int doc_id) {
  for (int j = 0; j < inner_num_iters; j++) {
    //compute gradient
    vec_mat_mul(tmp_V, doc_tpc[doc_id], tpc_word, K, V);
    vec_vec_add(tmp_V, doc_word[doc_id], -1, V);
    mat_vec_mul(tmp_K, tmp_V, tpc_word, K, V);
    //do gradient descent
    vec_vec_add(doc_tpc[doc_id], tmp_K, -lr, K);
    //proejection
    /*float sum=0;
    for(int i=0;i<K;i++) {
    	std::cout<<doc_tpc[doc_id][i]<<" ";
    	sum+=doc_tpc[doc_id][i];
    }
    std::cout<<std::endl;
    std::cout<<sum<<std::endl;
    std::cout<<std::endl;*/
    project_prob_smp(doc_tpc[doc_id], K, 1.0, MIN_BETA , tmp_K);
    /*sum=0;
    for(int i=0;i<K;i++){
    	std::cout<<doc_tpc[doc_id][i]<<" ";
    	sum+=doc_tpc[doc_id][i];
    }
    std::cout<<std::endl;
    std::cout<<sum<<std::endl;*/
  }
}

tm_mf::~tm_mf() {
  delete[] tmp_V;
  delete[] tmp_K;
  for(int i=0;i<K;i++)
    delete[] grad[i];
  delete []grad;
}

tm_mf::tm_mf(paras prs) {
  this->D=prs.D;
  this->inner_num_iters=prs.inner_num_iters;
  this->K=prs.K;
  this->lr=prs.lr;
  this->mb_size=prs.mb_size;
  this->num_epochs=prs.num_epochs;
  this->V=prs.V;
  this->num_eval=prs.num_eval;
  
  init_paras();
  load_data(prs.file_docs);
  
  mb=new int[mb_size];
  eval_mb=new int[num_eval];
  tmp_V=new float[V];
  tmp_K=new float[K];
  grad=new float*[K];
  for(int i=0;i<K;i++){
    grad[i]=new float[V];
  }
}

void tm_mf::load_data(char * file_docs) {
  load_mat(doc_word, D, V, file_docs);
}

void tm_mf::init_paras() {
  doc_word=new float*[D];
  for(int d=0;d<D;d++)
 	  doc_word[d]=new float[V];
  srand(1);
  doc_tpc=new float*[D];
  for(int d=0;d<D;d++){
    doc_tpc[d]=new float[K];
    rand_init_smp_vec(doc_tpc[d],K);
    for(int i=0;i<5;i++)
      std::cout<<doc_tpc[d][i]<<" ";
    std::cout<<std::endl;
  }
  
  tpc_word=new float*[K];
  for(int k=0;k<K;k++){
    tpc_word[k]=new float[V];
    rand_init_smp_vec(tpc_word[k],V);
  }
}

void tm_mf::learn() {
  int num_iters=num_epochs*D/mb_size;
  for(int i=0;i<num_iters;i++){
    // sample a mini-batch
    smp_mb(mb, mb_size, D);
    //perform inference
    infer();
    //perform parameter update
    update_tpc_word();
    //evaluate objective function
    evaluate();
  }
}

void tm_mf::update_tpc_word() {
  for(int i=0;i<K;i++) {
    memset(grad[i],0, sizeof(float)*V);
  }
  for(int iter=0;iter<inner_num_iters;iter++) {
    // compute gradient
    for(int i=0;i<mb_size;i++){
      int doc_id=mb[i];
      vec_mat_mul(tmp_V, doc_tpc[doc_id], tpc_word, K, V);
      vec_vec_add(tmp_V, doc_word[doc_id], -1, V);
      mat_rank1_update(grad, doc_tpc[doc_id], tmp_V, K, V);
    }
    // do gradient descent
    float coeff=-lr/mb_size;
    mat_mat_add(tpc_word, grad,coeff, K, V);
    
    for(int k=0;k<K;k++)
      project_prob_smp( tpc_word[k], V, 1.0, MIN_BETA , tmp_V);
    // projection
  }
}

