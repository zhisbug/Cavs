#ifndef CAVS_BACKEND_FUNCTOR_SORT_SCAN_CUH_
#define CAVS_BACKEND_FUNCTOR_SORT_SCAN_CUH_

namespace backend {

template<typename T>
__device__ inline void Comparator(T& valA, T& valB, bool direction) {
  if ((valA > valB) == direction) {
    T tmp; 
    tmp = valA;
    valA = valB;
    valB = tmp;
  }
}

template<typename T, unsigned SHARE_SIZE_LIMIT>
__global__ void BatchedMergeSort(T *inout, unsigned int batch, unsigned int N, bool direction) {
  __shared__ T s_val[SHARE_SIZE_LIMIT];
  T* d_val = inout + blockIdx.x*N+ threadIdx.x;
  s_val[threadIdx.x] = d_val[0];
  s_val[threadIdx.x+N/2] = d_val[N/2];

  for (unsigned size = 2; size <= N; size <<= 1) {
    unsigned stride = size / 2; 
    unsigned offset = threadIdx.x & (stride -1);
    __syncthreads();
    {
      unsigned pos = 2*threadIdx.x - offset; 
      Comparator(s_val[pos], s_val[pos+stride], direction);
      stride >>= 1;
    }
    for (; stride > 0; stride >>= 1) {
      __syncthreads(); 
      unsigned pos = 2*threadIdx.x - (threadIdx.x&(stride-1));
      if (offset >= stride) {
        Comparator(s_val[pos-stride], s_val[pos], direction);
      }
    }
  }
  __syncthreads();
  d_val[0] = s_val[threadIdx.x];
  d_val[N/2] = s_val[threadIdx.x+(N)/2];
}

} //namespace backend

#endif
