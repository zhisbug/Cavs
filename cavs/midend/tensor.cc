#include "cavs/midend/tensor.h"
#include "cavs/midend/types.h"
#include "cavs/util/logging.h"

namespace midend {

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
    data_ = alloc->Allocate<T>(elem_);   
  }
  ~TensorBuffer() override { alloc_->Deallocate<T>(data_); }
  FORCE_INLINE void* data() const override { return data_; }
  FORCE_INLINE size_t size() const override { return elem_*sizeof(T); }
  FORCE_INLINE size_t count() const override { return elem_; }

 private:
  T* data_;
  int elem_;

  DISALLOW_COPY_AND_ASSIGN(TensorBuffer);
};

Tensor::Tensor(const string& name, Allocator *a, 
        DataType type, const TensorShape& shape) 
    : name_(name), buf_(nullptr), shape_(nullptr), type_(type) {
  shape_.reset(new TensorShape(shape));
  Rebase(a, type, shape);
}

Tensor::Tensor(const string& name, Allocator *a, 
        DataType type, TensorShape&& shape) 
    : name_(name), buf_(nullptr), shape_(nullptr), type_(type) {
  shape_.reset(new TensorShape(std::move(shape)));
  Rebase(a, type, shape);
}

void Tensor::Rebase(Allocator *a, 
        DataType type, const TensorShape& shape) {
  shape_.reset(new TensorShape(shape));
  CASES(type, buf_.reset(new TensorBuffer<T>(a, shape_->n_elements())));
}

void Tensor::Rebase(Allocator *a, const Tensor& t) {
  Rebase(a, t.type_, *(t.shape_));
}

} //namespace midend
