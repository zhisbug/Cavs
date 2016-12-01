#include "cavs/midend/tensor.h"
#include "cavs/midend/types.h"
#include "cavs/util/logging.h"

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
  ~TensorBuffer() override { alloc_->Deallocate<T>(data_); }
  void* data() const override { return data_; }
  //size_t size() const override { return elem_*sizeof(T); }
  size_t count() const override { return elem_; }

 private:
  T* data_;
  int elem_;

  DISALLOW_COPY_AND_ASSIGN(TensorBuffer);
};

Tensor::Tensor(const string& name, Allocator *a, 
        DataType type, const TensorShape& shape) 
    : name_(name), buf_(nullptr), shape_(shape) {
  Reshape(a, type, shape);
}

void Tensor::Reshape(Allocator *a, 
        DataType type, const TensorShape& shape) {
  shape_ = shape;
  CASES(type, buf_.reset(new TensorBuffer<T>(a, shape_.n_elements())));
}

TensorShape::TensorShape(const ListValue& shape) {
  CHECK(shape.i_size() > 0);
  shape_.resize(shape.i_size());
  n_elements_ = 1;    
  for (int idx = 0; idx < shape.i_size(); idx++) {
    shape_[idx] = shape.i(idx);
    n_elements_ *= shape.i(idx); 
  }
}

TensorShape::TensorShape(std::initializer_list<int> shape) 
    : shape_(shape) {
  n_elements_ = 1;    
  for (const int dim : shape) {
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
