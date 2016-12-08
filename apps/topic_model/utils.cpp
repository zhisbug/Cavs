#include "utils.h"

#include <vector>

using namespace std;

void project_prob_smp(float *beta, const int &nTerms, 
      const float &dZ, const float &epsilon, float * mu_) {
  vector<int> U(nTerms);
  for ( int i=0; i<nTerms; i++ ) {
    mu_[i] = beta[i] - epsilon;
    U[i] = i + 1;
  }
  float dZVal = dZ - epsilon * nTerms; // make sure dZVal > 0
  
  /* project to a simplex. */
  float s = 0;
  int p = 0;
  while ( !U.empty() ) {
    int nSize = U.size();
    int k = U[ rand()%nSize ];
    
    /* partition U. */
    vector<int> G, L;
    int deltaP = 0;
    float deltaS = 0;
    for ( int i = 0; i < nSize; i++ ) {
      int j = U[i];
      
      if ( mu_[j-1] >= mu_[k-1] ) {
        if ( j != k ) G.push_back( j );
        deltaP ++;
        deltaS += beta[j-1];
      } else L.push_back( j );
    }
    
    if ( s + deltaS - (p + deltaP) * mu_[k-1] < dZ ) {
      s += deltaS;
      p += deltaP;
      U = L;
    } else {
      U = G;
    }
  }
  
  float theta = (s - dZ) / p;
  for ( int i=0; i<nTerms; i++ ) {
    beta[i] = max((double)(mu_[i]) - theta, 0.0) + epsilon;
  }
}



float vec_sq(float * vec, int K) {
  float sum=0;
  for(int i=0;i<K;i++)
  	sum+=vec[i]*vec[i];
  return sum;
}


float randfloat() {
  return (rand()%1000)/1000.0;
}

void mat_rank1_update(float ** mat, float * vec1, float * vec2, int nrows, int ncols) {
  for(int i=0;i<nrows;i++) {
    for(int j=0;j<ncols;j++)
      mat[i][j]+=vec1[i]*vec2[j];
  }
}

void vec_mat_mul(float * vec_out, float * vec_in, float ** mat, int nrows, int ncols) {
  memset(vec_out, 0, sizeof(float)*ncols);
  for(int i=0;i<nrows;i++) {
    if(abs(vec_in[i])<1e-5)
      continue;
    for(int j=0;j<ncols;j++)
      vec_out[j]+=vec_in[i]*mat[i][j];
  }
}

void vec_vec_add(float * vec1, float * vec2, float coeff, int len) {
  for(int i=0;i<len;i++)
    vec1[i]+=coeff*vec2[i];
}

void mat_vec_mul(float * vec_out, float * vec_in, float ** mat, int nrows, int ncols) {
  for(int i=0;i<nrows;i++) {
    float sum=0;
    for(int j=0;j<ncols;j++)
      sum+=mat[i][j]*vec_in[j];
    vec_out[i]=sum;
  }
}

void mat_mat_add(float ** mat1, float ** mat2, float coeff, int nrows, int ncols) {
  for(int i=0;i<nrows;i++) {
    for(int j=0;j<ncols;j++)
      mat1[i][j]+=coeff*mat2[i][j];
  }
}


void smp_mb(int* mb, int mb_size, int range) {
  if (mb_size==range) {
    for (int i=0;i<mb_size;i++)
      mb[i]=i;
	}else {
    for (int i=0;i<mb_size;i++)
      mb[i]=rand()%range;
  }
}

void rand_init_smp_vec(float * a, int dim) {
  float sum=0;
  for(int i=0;i<dim;i++) {
    a[i]=randfloat();
    sum+=a[i];
  }
  for(int i=0;i<dim;i++)
    a[i]/=sum;
}

void load_mat(float ** mat, int nrows, int ncols, char * filename) {
  FILE *fp=fopen(filename,"rb");
  for(int i=0;i<nrows;i++) {
    fread(mat[i], sizeof(float),ncols, fp);
  }
  fclose(fp);
}
