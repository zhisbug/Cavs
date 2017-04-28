#ifndef CAVS_BACKEND_FUNCTOR_ELEMENTWISE_H_
#define CAVS_BACKEND_FUNCTOR_ELEMENTWISE_H_

#include "cavs/util/macros.h"

namespace backend {

namespace math {

template <typename T>
struct Abs {
  FORCE_INLINE __DEVICE__ static T Compute(T inp) {
    return abs(inp);
  }
};

template <typename T>
struct Square {
  FORCE_INLINE __DEVICE__ static T Compute(T inp) {
    return inp * inp;
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

template <typename T>
struct Max {
  FORCE_INLINE __DEVICE__ static T Compute(T inp0, T inp1) {
    return (inp0 > inp1) ? inp0 : inp1;
  }
};

template <typename T>
struct Min {
  FORCE_INLINE __DEVICE__ static T Compute(T inp0, T inp1) {
    return (inp0 < inp1) ? inp0 : inp1;
  }
};

template <typename T>
struct Equal {
  FORCE_INLINE __DEVICE__ static bool Compute(T inp0, T inp1) {
    return (inp0 == inp1);
  }
};

template <typename T>
struct Assign {
  FORCE_INLINE __DEVICE__ static T Compute(T inp) {
    return (inp);
  }
};

template <typename T, typename U>
struct Cast {
  FORCE_INLINE __DEVICE__ static T Compute(U inp) {
    return (T)(inp);
  }
};

} //namespace math

} //namespace backend

#endif
