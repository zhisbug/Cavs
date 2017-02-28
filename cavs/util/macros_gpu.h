#ifndef CAVS_UTIL_MACROS_GPU_H_
#define CAVS_UTIL_MACROS_GPU_H_

#include "cavs/util/logging.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define checkCudaError(status)                      \
    do {                                            \
      cudaError_t err = cudaGetLastError();         \
      if (err != cudaSuccess) {                     \
        LOG(FATAL) << "Last CUDA failure: "         \
                   << cudaGetErrorString(err);      \
      }                                             \
      if (status != cudaSuccess) {                  \
        LOG(FATAL) << "CUDA failure: "              \
                   << cudaGetErrorString(status);   \
      }                                             \
    }while(0)

#define checkCublasError(status)                    \
    do {                                            \
      cudaError_t err = cudaGetLastError();         \
      if (err != cudaSuccess) {                     \
        LOG(FATAL) << "Last CUDA failure: "         \
                   << cudaGetErrorString(err);      \
      }                                             \
      if (status != CUBLAS_STATUS_SUCCESS) {        \
        LOG(FATAL) << "CUDA failure: "              \
                   << status;                       \
      }                                             \
    }while(0)

#define checkCUDNNError(status)                     \
    do {                                            \
      cudaError_t err = cudaGetLastError();         \
      if (err != cudaSuccess) {                     \
        LOG(FATAL) << "Last CUDA failure: "         \
                   << cudaGetErrorString(err);      \
      }                                             \
      if (status != CUDNN_STATUS_SUCCESS){          \
        LOG(FATAL) << "CUDNN failure: "             \
                   << cudnnGetErrorString(status);  \
      }                                             \
    }while(0)


#endif
