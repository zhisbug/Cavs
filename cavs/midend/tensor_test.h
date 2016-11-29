#ifndef CAVS_MIDEND_TENSOR_TEST_H_
#define CAVS_MIDEND_TENSOR_TEST_H_

#include "cavs/midend/tensor.h"
#include "cavs/midend/macros_gpu.h"

namespace cavs {

namespace test {

//Just implement the GPU routines up to now

template <typename T>
void FillValues(Tensor* tensor, const vector<T>& vals) {
  CHECK_NOTNULL(tensor);
  T* buf = tensor->mutable_data<T>();
  CHECK(tensor->count() == vals.size());
  checkCudaError(cudaMemcpy(buf, vals.data(), vals.size()*sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void FetchValues(vector<T>& vals, const Tensor* tensor) {
  CHECK_NOTNULL(tensor);
  const T* buf = tensor->data<T>();
  CHECK_NOTNULL(buf);
  vals.resize(tensor->count());
  checkCudaError(cudaMemcpy(vals.data(), buf, vals.size()*sizeof(T), cudaMemcpyDeviceToHost));
}

} //namespace test

} //namespace cavs

#endif
