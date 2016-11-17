#ifndef ELEMENTWISE_OPS_H_
#define ELEMENTWISE_OPS_H_

#include "elementwise_ops_common.h"

namespace Functor {

template <typename op, typename T, typename R = T>
struct UnaryFunctor {
  virtual void operator () (R* out, const T *inp, int n) = 0;
};

template <typename op, typename T, typename R = T>
struct BinaryFunctor {
  virtual void operator () (R* out, const T *inp0, const T *inp1, int n) = 0;
};

}

namespace Math {

template <typename T, typename R = T>
struct Abs {
  FORCE_INLINE R operator () (T inp) {
    return abs(inp);
  }
};

template <typename T, typename R = T>
struct Add {
  FORCE_INLINE R operator () (T inp0, T inp1) {
    return (inp0 + inp1);
  }
};

}

#endif
