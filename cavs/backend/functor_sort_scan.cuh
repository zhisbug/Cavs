#ifndef CAVS_BACKEND_FUNCTOR_SORT_SCAN_CUH_
#define CAVS_BACKEND_FUNCTOR_SORT_SCAN_CUH_

#include <stdio.h>

namespace backend {

template <typename T>
__device__ inline void Comparator(T& valA, T& valB, bool direction) {
  if ((valA > valB) == direction) {
    T tmp; 
    tmp = valA;
    valA = valB;
    valB = tmp;
  }
}

template <typename T, unsigned int SHARE_SIZE_LIMIT>
__global__ void BatchedMergeSort(T* inout, unsigned int N, bool direction) {
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
  d_val[N/2] = s_val[threadIdx.x+N/2];
}

template <typename T, unsigned int SHARE_SIZE_LIMIT>
__global__ void BatchedScan(T* inout, unsigned int N) {}
//N == blockDim.x
//N < 1024 (warp_id < 32)
//There must be less than 32 warps in one block,
//as required in the syntax of CUDA)
/*template <unsigned int SHARE_SIZE_LIMIT>*/
template <unsigned int SHARE_SIZE_LIMIT>
__global__ void BatchedScan(float* inout, unsigned int N) {
  __shared__ float s_val[SHARE_SIZE_LIMIT]; 
  int id = threadIdx.x + blockIdx.x*N;
  const int warpSize = 1 << 5;
  int lane_id = threadIdx.x & (warpSize-1);
  int warp_id = threadIdx.x >> 5;
  float val = inout[id];
  #pragma unroll
  for (int i = 1; i < warpSize; i <<= 1) {
    float pre_sum = __shfl_up(val, i, warpSize);
    if (lane_id >= i) val += pre_sum;
  }
  s_val[threadIdx.x] = val;
  __syncthreads();
  
  /*printf("%d\t%d\t%f\n", threadIdx.x, lane_id, val);*/
  for (int i = 1; i <= (N-1) >> 5; i <<= 1) {
    if (warp_id >= i) {
      float pre_sum = s_val[((warp_id-i+1) << 5)-1];
      __syncthreads();
      s_val[threadIdx.x] += pre_sum;
      __syncthreads();
    }  
  }
  inout[id] = s_val[threadIdx.x];
}

} //namespace backend

#endif
