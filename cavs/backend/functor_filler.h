#ifndef CAVS_BACKEND_FUNCTOR_FILLER_H_
#define CAVS_BACKEND_FUNCTOR_FILLER_H_

#include "cavs/util/macros.h"
#include "cavs/util/macros_gpu.h"

#include <random>

namespace backend {

template <typename T>
struct UniformNormalizer {
  FORCE_INLINE static void Compute(T* buf, int N) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(0.f, 1.f);
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

template <typename T>
struct NormalRandom {
  FORCE_INLINE static void Compute(T* buf, int N) {
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(0.f, 1.f);
    for (unsigned i = 0; i < N; i++) {
      buf[i] = distribution(generator);
    }
  }
};

template <typename OP, typename T>
struct Filler {
  Filler(const OpDef& op_def) {
    stride_ = GetSingleArg<int>(op_def, "stride", 0);
    CHECK(stride_ >= 0);
  }

  FORCE_INLINE virtual void Compute(T* buf, int N) {
    if (stride_ > 0) {
      for (int i = 0; i < N; i+=stride_) {
        OP::Compute(buf+i, (i+stride_>N) ? (N-i) : stride_);
      }
    }else 
      OP::Compute(buf, N);
  }

  int stride_;
};

} //namespace backend

#endif

