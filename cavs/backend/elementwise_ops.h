#ifndef CAVS_KERNEL_ELEMENTWISE_OPS_H_
#define CAVS_KERNEL_ELEMENTWISE_OPS_H_

#include "cavs/backend/elementwise_ops_common.h"

namespace cavs {

//template <typename op, typename T>
//struct UnaryFunctor {
  //virtual void operator () (T* out, const T *inp, int n) = 0;
//};

//template <typename op, typename T>
//struct BinaryFunctor {
  //virtual void operator () (T* out, const T *inp0, const T *inp1, int n) = 0;
//};
namespace math {

template <typename T>
struct Abs {
  FORCE_INLINE __DEVICE__ static T Compute(T inp) {
    return abs(inp);
  }
};

template <typename T>
struct Add {
  FORCE_INLINE __DEVICE__ static T Compute(T inp0, T inp1) {
    return (inp0 + inp1);
  }
};

} //namespace math

} //namespace cavs

#endif
