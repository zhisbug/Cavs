#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <cstring>

float randfloat();
void mat_rank1_update(float ** mat, float * vec1, float * vec2, int nrows, int ncols);
void vec_mat_mul(float* vec_out, float * vec_in, float ** mat, int nrows, int ncols);
void vec_vec_add(float* vec1, float * vec2, float coeff, int len);
void mat_vec_mul(float* vec_out, float * vec_in, float ** mat, int nrows, int ncols);
void mat_mat_add(float** mat1, float ** mat2, float coeff, int nrows, int ncols);

void smp_mb(int* mb, int mb_size, int range);
void rand_init_smp_vec(float * a, int dim);
void load_mat(float ** mat, int nrows, int ncols, char * filename);
float vec_sq(float * vec, int K);
void project_prob_smp(float *beta, const int &nTerms, const float &dZ, const float &epsilon, float * mu_ );

#endif
