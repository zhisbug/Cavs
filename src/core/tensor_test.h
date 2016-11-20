#ifndef TENSOR_TEST_H_
#define TENSOR_TEST_H_

namespace test {

bool InsertTensor(Tensor* t, Session* s) {
  return s->InsertTensor(t->name(), t); 
}

template <typename T>
void FillValues(Tensor* tensor, vector<T>& vals) {
  T* buf = tensor->data<T>();
  CHECK(tensor->count(), vals.size());
  checkCudaError(cudaMemcpy(buf, vals.data(), vals.size()*sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void FetchValues(vector<T>& vals, Tensor* tensor) {
  T* buf = tensor->data<T>();
  vals.resize(tensor->count());
  checkCudaError(cudaMemcpy(vals.data(), buf, vals.size()*sizeof(T), cudaMemcpyDeviceToHost));
}

}

#endif
