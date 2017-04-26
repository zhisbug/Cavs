#ifndef CAVS_BACKEND_FUNCTOR_FUNCTOR_REDUCTION_CUH_
#define CAVS_BACKEND_FUNCTOR_FUNCTOR_REDUCTION_CUH_

#include "cavs/util/macros_gpu.h"

namespace backend {

template <typename T>
__inline__ __device__
T warpReduceMax(T& val, int &idx) {
  const int warpSize = 32;
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    T r_value = __shfl_down(val, offset);
    int r_idx = __shfl_down(idx, offset);
    if (r_value > val) {
      val = r_value;
      idx = r_idx;
    }
  }
  return val;
}

template <typename T>
__global__ void BatchedArgmaxKernel(T *out, int *index, const T* inp, int N) {
  __shared__ int index_buf[32];
  __shared__ T value_buf[32];
  const int warpSize = 32;
  int lane = threadIdx.x%warpSize;
  int warp_id = threadIdx.x/warpSize;
  int offset = blockIdx.x*N;
  int curr_index = threadIdx.x;
  /*T curr_max = std::numeric_limits<int>::min();*/
  T flag = inp[offset+0];
  T curr_max = flag;
  for (int round = 0; round < (N+blockDim.x-1)/blockDim.x; round++) {
    int offset_within_vec = threadIdx.x + round*blockDim.x;
    int idx = offset + offset_within_vec;
    if (offset_within_vec < N) {
      if (inp[idx] > curr_max) {
        curr_index = offset_within_vec;
        curr_max = inp[idx];
      }
    }
  }

  /*printf("%d\t%f\t%d\n", threadIdx.x, curr_max, curr_index);*/
  warpReduceMax(curr_max, curr_index);
  if (lane == 0) {
    index_buf[warp_id] = curr_index;
    value_buf[warp_id] = curr_max;
  }
  __syncthreads();
  curr_index = (threadIdx.x < (blockDim.x+warpSize-1) / warpSize) ?
    index_buf[lane] : threadIdx.x;
  curr_max = (threadIdx.x < (blockDim.x+warpSize-1) / warpSize) ?
    value_buf[lane] : flag;
  if (warp_id == 0) {
    warpReduceMax(curr_max, curr_index);
  }
  if (threadIdx.x == 0) {
    out[blockIdx.x] = curr_max;
    index[blockIdx.x] = curr_index;
    /*printf("%f\t%d\n", curr_max, curr_index);*/
  }
}

template <typename T>
void BatchedArgmax(T* out, int* index, const T* in, int N, int Batch) {
  const int MAX_THREADS_IN_BLOCK = 1 << 10;
  unsigned int threadsPerBlock =
      (MAX_THREADS_IN_BLOCK > N)? N : MAX_THREADS_IN_BLOCK;
  unsigned int blocksPerGrid = Batch;

  BatchedArgmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(out, index, in, N);
  checkCudaError(cudaGetLastError());
}

}

#endif
