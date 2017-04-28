#ifndef CAVS_BACKEND_FUNCTOR_FILLER_H_
#define CAVS_BACKEND_FUNCTOR_FILLER_H_

#include "cavs/util/macros.h"
#include "cavs/util/macros_gpu.h"

#include <random>
#include <boost/random.hpp>
#include <boost/math/special_functions/next.hpp>

namespace backend {

template <typename T>
struct Xavier {
  FORCE_INLINE static void Compute(T* buf, int N) {
    float scale = sqrt(3.f/N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-scale, scale);
    for (unsigned i = 0; i < N; i++) {
      buf[i] = distribution(generator);
    }

    //typedef boost::mt19937 rng_t;
    //boost::variate_generator<rng_t*,boost::uniform_real<float> > var_uniform(
        //new rng_t(1), boost::uniform_real<float>(-scale,
          //boost::math::nextafter<float>(scale, std::numeric_limits<float>::max())));
    //boost::uniform_real<float> random_distribution(-scale,
    //boost::math::nextafter<float>(scale, std::numeric_limits<float>::max()));
    //boost::variate_generator<rng_t*, boost::uniform_real<float> >
        //variate_generator(new rng_t(1), random_distribution);
    //for (unsigned i = 0; i < N; i++) {
      //buf[i] = variate_generator();
    //}
  }
};

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
    CHECK(op_def.output_size() == 1);
    CHECK(op_def.shape(0).dim_size() >= 1);
    int default_stride = 1;
    for (int i = 1; i < op_def.shape(0).dim_size(); i++)
      default_stride *= op_def.shape(0).dim(i);
    stride_ = GetSingleArg<int>(op_def, "stride", default_stride);
    CHECK(stride_ >= 0);
  }
  FORCE_INLINE virtual void Compute(T* buf, int N) {
    if (stride_ > 0) {
      for (int i = 0; i < N; i += stride_) {
        OP::Compute(buf+i, (i+stride_>N) ? (N-i) : stride_);
      }
    }else 
      OP::Compute(buf, N);
  }
  int stride_;
};

} //namespace backend

#endif

