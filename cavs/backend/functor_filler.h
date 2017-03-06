#ifndef CAVS_BACKEND_FUNCTORS_ELEMENTWISE_H_
#define CAVS_BACKEND_FUNCTORS_ELEMENTWISE_H_

#include "cavs/util/macros.h"
#include "cavs/util/macros_gpu.h"

#include <random>

namespace backend {

template <typename T>
struct UniformNormalizer {
  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution(0.f, 1.f);
  FORCE_INLINE static void Compute(T* buf, int N) {
    T sum = 0;
    for (unsigned i = 0; i < N; i++) {
      buf[i] = distribution(generator);
      sum += buf[i];
    }
    for (unsigned i = 0; i < N; i++) {
      buf[i] /= sum;
    }
  }
};

template <typename Op, typename T>
struct CudaFiller {
  CudaFiller(const OpDef& op_def) {}
  FORCE_INLINE void Compute(T* buf, int N) {
    vector<T> cpu_buf(N);
    for (int i = 0; i < N; i+=stride) {
      Op<T>.Compute(cpu_buf.data()+i, (i+stride>=N) ? (N-i) : stride);
    }
    checkCudaError(cudaMemcpy(buf, cpu_buf.data(), N*sizeof(T),
                              cudaMemcpyHostToDevice));
  }
};

template <typename Op, typename T>
struct CudaConstantFiller {
  CudaConstantFiller(const OpDef& op_def) {}
  FORCE_INLINE void Compute(T* buf, int N) {
    UnaryConstScalarKernel<OP, T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>>(
        buf, value, n);
  }
  T value;
};

} //namespace backend

#endif

