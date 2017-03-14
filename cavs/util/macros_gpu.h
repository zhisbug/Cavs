#ifndef CAVS_UTIL_MACROS_GPU_H_
#define CAVS_UTIL_MACROS_GPU_H_

#include "cavs/util/logging.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define checkCudaError(stmt)                     \
    do {                                         \
      cudaError_t err = (stmt);                  \
      if (err != cudaSuccess) {                  \
        LOG(FATAL) << "CUDA failure: "           \
                   << cudaGetErrorString(err);   \
      }                                          \
    }while(0)

#define checkCublasError(stmt)                   \
    do {                                         \
      cublasStatus_t err = (stmt);               \
      if (err != CUBLAS_STATUS_SUCCESS) {        \
        LOG(FATAL) << "CUDA failure: "           \
                   << err;                       \
      }                                          \
    }while(0)

#define checkCUDNNError(stmt)                    \
    do {                                         \
      cudnnStatus_t err = (stmt);                \
      if (err != CUDNN_STATUS_SUCCESS){          \
        LOG(FATAL) << "CUDNN failure: "          \
                   << cudnnGetErrorString(err);  \
      }                                          \
    }while(0)


#endif
