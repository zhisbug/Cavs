#ifndef CAVS_MIDEND_MACROS_GPU_H_
#define CAVS_MIDEND_MACROS_GPU_H_

#include "macros.h"
#include "cavs/util/logging.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define checkCublasError(status)                             \
        do {                                                 \
            if (status != CUBLAS_STATUS_SUCCESS) {           \
                LOG(FATAL) << "CUDA failure: "               \
                           << status;                        \
            }                                                \
        }while(0)

#define checkCudaError(status)                               \
        do {                                                 \
            if (status != cudaSuccess) {                     \
                LOG(FATAL) << "CUDA failure: "               \
                           << cudaGetErrorString(status);    \
            }                                                \
       }while(0)

#define CUDA_1D_KERNEL_LOOP(i, n)                           \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;     \
            i < (n); i += blockDim.x * gridDim.x)

const int THREADS_PER_BLOCK = 512;
FORCE_INLINE int BLOCKS_PER_GRID(const int N) {
    return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

#endif
