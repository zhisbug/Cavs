#ifndef CAVS_BACKEND_ELEMENTWISE_OPS_H_
#define CAVS_BACKEND_ELEMENTWISE_OPS_H_

#include "cavs/backend/elementwise_ops_common.h"

namespace backend {

namespace math {

template <typename T>
struct Abs {
  FORCE_INLINE __DEVICE__ static T Compute(T inp) {
    return abs(inp);
  }
};

template <typename T>
struct Neg {
  FORCE_INLINE __DEVICE__ static T Compute(T inp) {
    return -inp;
  }
};

template <typename T>
struct Add {
  FORCE_INLINE __DEVICE__ static T Compute(T inp0, T inp1) {
    return (inp0 + inp1);
  }
};

template <typename T>
struct Sub {
  FORCE_INLINE __DEVICE__ static T Compute(T inp0, T inp1) {
    return (inp0 - inp1);
  }
};

template <typename T>
struct Mul {
  FORCE_INLINE __DEVICE__ static T Compute(T inp0, T inp1) {
    return (inp0 * inp1);
  }
};

template <typename T>
struct Div {
  FORCE_INLINE __DEVICE__ static T Compute(T inp0, T inp1) {
    return (inp0 / inp1);
  }
};

} //namespace math

} //namespace backend

#endif
