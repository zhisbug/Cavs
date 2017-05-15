#ifndef CAVS_BACKEND_FUNCTOR_FILLER_H_
#define CAVS_BACKEND_FUNCTOR_FILLER_H_

#include "cavs/util/macros.h"
#include "cavs/util/macros_gpu.h"

#include <random>
#include <boost/random.hpp>
#include <boost/math/special_functions/next.hpp>

namespace backend {

template <typename T>
struct Filler {
  Filler(const OpDef& op_def) : op_def_(op_def) {
    CHECK(op_def.output_size() == 1);
    CHECK(op_def.shape(0).dim_size() >= 1);
    stride_ = GetSingleArg<int>(op_def, "stride", 0);
    CHECK(stride_ >= 0);
  }
  virtual void FillRaw(T* buf, int N) = 0;

  void Compute(T* buf, int N) {
    int stride  = (stride_ == 0)? N : stride_;
      for (int i = 0; i < N; i += stride) {
        FillRaw(buf+i, (i+stride>N) ? (N-i) : stride);
      }
  }

 protected:
  int stride_;
  OpDef op_def_;//debug
};

template <typename T>
struct ConstantFiller : Filler<T> {
  ConstantFiller(const OpDef& op_def) : Filler<T>(op_def) {
    value_ = GetSingleArg<T>(op_def, "const_value");
  }
  virtual void FillRaw(T* buf, int N) override {
    for (unsigned i = 0; i < N; i++) {
      buf[i] = value_;
    }
  }

 private:
  T value_;
};


template <typename T>
struct Xavier : Filler<T> {
  Xavier(const OpDef& op_def) : Filler<T>(op_def) {
    CHECK(op_def.output_size() == 1);
    CHECK(op_def.shape(0).dim_size() >= 1);
    int N = 1;
    for (int i = 1; i < op_def.shape(0).dim_size(); i++)
      N *= op_def.shape(0).dim(i);
    scale_ = sqrt(3.f/N);
  }
  virtual void FillRaw(T* buf, int N) override {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-scale_, scale_);
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

 private:
  T scale_;
};

template <typename T>
struct UniformRandom : Filler<T> {
  UniformRandom(const OpDef& op_def) : Filler<T>(op_def) {
    minval_ = GetSingleArg<float>(op_def, "minval");
    maxval_ = GetSingleArg<float>(op_def, "maxval");
    CHECK(minval_ < maxval_);
  }
  virtual void FillRaw(T* buf, int N) override {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(minval_, maxval_);
    for (unsigned i = 0; i < N; i++) {
      buf[i] = distribution(generator);
    }
    //typedef boost::mt19937 rng_t;
    //boost::uniform_real<float> random_distribution(minval_, maxval_);
    //boost::variate_generator<rng_t*, boost::uniform_real<float> >
        //variate_generator(new rng_t(1), random_distribution);
    //for (unsigned i = 0; i < N; i++) {
      //buf[i] = variate_generator();
    //}
  }

 protected:
  float minval_;
  float maxval_;
};

template <typename T>
struct NormalRandom : Filler<T> {
  NormalRandom(const OpDef& op_def) : Filler<T>(op_def) {}
  virtual void FillRaw(T* buf, int N) override {
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(0.f, 1.f);
    for (unsigned i = 0; i < N; i++) {
      buf[i] = distribution(generator);
    }
  }
};

template <typename T>
struct UniformRandomNormalized : UniformRandom<T> {
  UniformRandomNormalized(const OpDef& op_def) : UniformRandom<T>(op_def) {}
  virtual void FillRaw(T* buf, int N) override {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(this->minval_, this->maxval_);
    T sum = 0;
    for (unsigned i = 0; i < N; i++) {
      buf[i] = distribution(generator);
      sum += buf[i];
    }
    CHECK(sum != 0);
    for (unsigned i = 0; i < N; i++) {
      buf[i] /= sum;
    }
  }
};

} //namespace backend

#endif

