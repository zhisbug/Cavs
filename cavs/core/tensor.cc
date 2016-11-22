#include "cavs/core/tensor.h"
#include "cavs/core/logging.h"
#include "cavs/core/types.h"

namespace cavs {

#define CASE(TYPE, STMTS)                             \
  case DataTypeToEnum<TYPE>::value: {                 \
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
      LOG(FATAL) << "Unsupported type:" << TYPE_ENUM; \
      break;                                          \
  }

template <typename T>
class TensorBuffer : public TensorBufferBase {
 public:
  TensorBuffer(Allocator* alloc, size_t elem) 
      : TensorBufferBase(alloc), elem_(elem) {
    data_ = alloc->Allocate<T>(elem_) ;   
  }
  void* data() const override { return data_; }
  //size_t size() const override { return elem_*sizeof(T); }
  size_t count() const override { return elem_; }

 private:
  T* data_;
  int elem_;

  DISALLOW_COPY_AND_ASSIGN(TensorBuffer);
};

Tensor::Tensor(const string& name, Allocator *a, DataType type, const TensorShape& shape) 
    : name_(name) {
  CHECK(RealAllocate(a, type, shape));
}

bool Tensor::RealAllocate(Allocator *a, DataType type, const TensorShape& shape){
  if (buf_)
    return false;
  shape_ = shape;
  CASES(type, buf_ = new TensorBuffer<T>(a, shape_.n_elements()));
  return true;
}

TensorShape::TensorShape(vector<int>& shape)
    : shape_(shape) {
  n_elements_ = 1;    
  for (const int dim : shape_) {
    n_elements_ *= dim; 
  }
}

TensorShape::TensorShape(std::initializer_list<int> shape) 
    : shape_(shape) {
  n_elements_ = 1;    
  for (const int dim : shape_) {
    n_elements_ *= dim; 
  }
}

void TensorShape::set_dim(int d, int size) {
  CHECK(dims() >= d);
  shape_[d] = size;
}

void TensorShape::add_dim(int size) {
  shape_[dims()] = size;
}

} //namespace cavs
