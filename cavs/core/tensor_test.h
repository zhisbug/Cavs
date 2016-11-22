#ifndef CAVS_TENSOR_TEST_H_
#define CAVS_TENSOR_TEST_H_

#include "cavs/core/tensor.h"
#include "cavs/core/macros_gpu.h"

namespace cavs {

namespace test {

//Just implement the GPU routines up to now

template <typename T>
void FillValues(Tensor* tensor, const vector<T>& vals) {
  T* buf = tensor->mutable_data<T>();
  CHECK(tensor->count() == vals.size());
  checkCudaError(cudaMemcpy(buf, vals.data(), vals.size()*sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void FetchValues(vector<T>& vals, const Tensor* tensor) {
  const T* buf = tensor->data<T>();
  vals.resize(tensor->count());
  checkCudaError(cudaMemcpy(vals.data(), buf, vals.size()*sizeof(T), cudaMemcpyDeviceToHost));
}

} //namespace test

} //namespace cavs

#endif
