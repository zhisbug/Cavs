#include "tensor.h"

#define CASE(TYPE, STMTS)                             \
  case DataTypeToEnum<TYPE>::value                    \
    typedef TYPE T;                                   \
    STMTS;                                            \
    break;                                            \
  }

#define CASES(TYPE_ENUM, STMTS)                       \
  switch (TYPE_ENUM) {                                \
    CASE(float, STMTS)                                \
    CASE(double, STMTS)                               \
    CASE(int, STMTS)                                  \
    default:                                          \
      LOG(FATAL) << "Unsupported type:" << TYPE_ENUM  \
      break;                                          \
  }

template <typename T>
class TensorBuffer : public TensorBufferBase {
 public:
  TensorBuffer(Allocator* alloc);
  void* data() const override { return data_; }
  size_t size() const override { return elem_*sizeof(T); }

 private:
  T* data_;
  int elem_;

  DISALLOW_COPY_AND_ASSIGN(TensorBuffer);
};

Tensor::Tensor(Allocator *a, DataType type, const TensorShape& shape)
    : shape_(shape) {
  CASES(buf_ = new TensorBuffer<T>(shape_.num_elements()))
}

TensorShape::TensorShape(vector<int>&& shape)
    : shape_(std::move(shape)) {
  num_elements_ = 1;    
  for (int i = 0; i < dims(); i++) {
    num_elements_ *= shape_[i]; 
  }
}

void TensorShape::set_dim(int d, int size) {
  CHECK(dims() >= d);
  shape_[d] = size;
}

void TensorShape::add_dim(int size) {
  shape_[dims()] = size;
}
