#ifndef CAVS_BACKEND_FUNCTOR_BATCHED_MEMCPY_CUH_
#define CAVS_BACKEND_FUNCTOR_BATCHED_MEMCPY_CUH_

#include "cavs/backend/cuda_common.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

template <typename T>
__global__ void ContinuousMemcpyKernel(
    T *out, const T* inp, int n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = inp[i];
  }
}

//we assume the GridDim.x = dynamic dimension
template <typename T>
__global__ void BatchedDynamicSliceCopyKernel(
    T *out, int out_stride, const T* inp, int inp_stride, int copy_length) {
  /*int inp_offset = blockIdx.x*inp_stride + threadIdx.x;*/
  /*int out_offset = blockIdx.x*out_stride + threadIdx.x;*/
  /*if (threadIdx.x < copy_length) {*/
    /*out[out_offset] = inp[inp_offset];*/
  /*}*/
  int inp_offset = blockIdx.x*inp_stride;
  int out_offset = blockIdx.x*out_stride;
  for (int tid = threadIdx.x; tid < copy_length; tid += blockDim.x) {
    out[out_offset + tid] = inp[inp_offset + tid];
  }
}

template <typename T>
__global__ void BatchedDynamicSelectedInputSliceCopyKernel(
    T *out, int out_stride, const T* inp, int inp_stride, const int* ids, int copy_length) {
  /*int inp_offset = ids[blockIdx.x]*inp_stride + threadIdx.x;*/
  /*int out_offset = blockIdx.x*out_stride + threadIdx.x;*/
  /*if (threadIdx.x < copy_length) {*/
    /*out[out_offset] = inp[inp_offset];*/
  /*}*/
  int inp_offset = ids[blockIdx.x]*inp_stride;
  int out_offset = blockIdx.x*out_stride;
  for (int tid = threadIdx.x; tid < copy_length; tid += blockDim.x) {
    out[out_offset + tid] = inp[inp_offset + tid];
  }
}

template <typename T>
__global__ void BatchedDynamicSelectedOutputSliceCopyKernel(
    T *out, int out_stride, const int* ids, const T* inp, int inp_stride, int copy_length) {
  /*int inp_offset = blockIdx.x*inp_stride + threadIdx.x;*/
  /*int out_offset = ids[blockIdx.x]*out_stride + threadIdx.x;*/
  /*if (threadIdx.x < copy_length) {*/
    /*out[out_offset] = inp[inp_offset];*/
  /*}*/
  int inp_offset = blockIdx.x*inp_stride;
  int out_offset = ids[blockIdx.x]*out_stride;
  for (int tid = threadIdx.x; tid < copy_length; tid += blockDim.x) {
    out[out_offset + tid] = inp[inp_offset + tid];
  }
}

template <typename T>
__global__ void BatchedDynamicSelectedAssignZeroKernel(
    T *out, int out_stride, const int* ids, int copy_length) {
  /*int inp_offset = ids[blockIdx.x]*inp_stride + threadIdx.x;*/
  /*int out_offset = blockIdx.x*out_stride + threadIdx.x;*/
  /*if (threadIdx.x < copy_length) {*/
    /*out[out_offset] = inp[inp_offset];*/
  /*}*/
  int out_offset = blockIdx.x*out_stride;
  for (int tid = threadIdx.x; tid < copy_length; tid += blockDim.x) {
    out[out_offset + tid] = 0;
  }
}

} //namespace backend

#endif

