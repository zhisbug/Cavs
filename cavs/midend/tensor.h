#ifndef CAVS_MIDEND_TENSOR_H_
#define CAVS_MIDEND_TENSOR_H_

#include "cavs/midend/types.pb.h"
#include "cavs/midend/devices.pb.h"
#include "cavs/midend/devices.h"
#include "cavs/midend/allocator.h"
#include "cavs/util/logging.h"
#include "cavs/util/macros.h"

#include <vector>
#include <string>
#include <memory>

using std::vector;
using std::string;

namespace midend {

//data
class TensorBufferBase {
 public:
  TensorBufferBase(Allocator* alloc) : alloc_(alloc) {}
  FORCE_INLINE DeviceType device_type() const { return alloc_->type(); }
  virtual ~TensorBufferBase() {}
  virtual void* data() const = 0;
  virtual size_t size() const = 0;
  virtual size_t count() const = 0;

 protected:
  Allocator* const alloc_;
};

//metadata
class TensorShape {
 public:
  TensorShape() : n_elements_(0) {}
  explicit TensorShape(const TensorShapeDef& shape);
  explicit TensorShape(std::initializer_list<int> shape);
  explicit TensorShape(const TensorShape& shape);
  explicit TensorShape(TensorShape&& shape);
  TensorShape& operator =(const TensorShape& b);
  FORCE_INLINE int n_elements() const { return n_elements_; }
  FORCE_INLINE int dims() const { return shape_.size(); }
  FORCE_INLINE int dims(int idx) const { return shape_.at(idx); }
  void set_dim(int d, int size);
  void add_dim(int size);

 private:
  vector<int> shape_;
  int n_elements_;
};

class TensorCApi;
class Tensor {
 public:
  Tensor() {}
  Tensor(const string& name, Allocator *a, DataType type, const TensorShape& shape);
  Tensor(const string& name, Allocator *a, DataType type, TensorShape&& shape);
  Tensor(const Tensor& t) { *this = t; }
  FORCE_INLINE Tensor& operator =(const Tensor& tensor) {
    buf_   = tensor.buf_;
    shape_ = tensor.shape_;
    name_  = tensor.name_;
    type_  = tensor.type_;
    return *this;
  }
  FORCE_INLINE DeviceType device_type() const { return buf_->device_type(); }
  FORCE_INLINE const string& name() const { return name_; }
  //for opeators
  FORCE_INLINE size_t count() const { return buf_->count(); }
  FORCE_INLINE int dims() const { return shape_->dims(); }
  FORCE_INLINE int dims(int idx) const { return shape_->dims(idx); }

  void Rebase(Allocator *a, DataType type, const TensorShape& shape);
  void Rebase(Allocator *a, const Tensor& t);
  template <typename T>
    T* mutable_data() const { return reinterpret_cast<T*>(buf_->data()); }
  template <typename T>
    const T* data() const { return reinterpret_cast<T*>(buf_->data()); }

  friend class TensorCApi;
  friend class DeviceContext;

 private:
  std::shared_ptr<TensorBufferBase> buf_;
  std::shared_ptr<TensorShape> shape_;
  string name_;
  DataType type_;
};

FORCE_INLINE TensorShape::TensorShape(const TensorShapeDef& shape) {
  CHECK(shape.dim_size() > 0);
  shape_.resize(shape.dim_size());
  n_elements_ = 1;    
  for (int idx = 0; idx < shape.dim_size(); idx++) {
    shape_[idx] = shape.dim(idx);
    n_elements_ *= shape.dim(idx); 
  }
}

//mainly for test usage
FORCE_INLINE TensorShape::TensorShape(std::initializer_list<int> shape) 
    : shape_(shape) {
  n_elements_ = 1;    
  for (const int dim : shape) {
    n_elements_ *= dim; 
  }
}

FORCE_INLINE TensorShape::TensorShape(const TensorShape& shape) {
  *this = shape;
}

FORCE_INLINE TensorShape::TensorShape(TensorShape&& shape) {
  n_elements_ = shape.n_elements_;
  shape_ = std::move(shape.shape_);
}

FORCE_INLINE TensorShape& TensorShape::operator =(const TensorShape& b) {
  n_elements_ = b.n_elements_;
  shape_ = b.shape_;
  return *this;
}

FORCE_INLINE void TensorShape::set_dim(int d, int size) {
  CHECK(dims() >= d);
  n_elements_ = n_elements_/shape_[d]*size;
  shape_[d] = size;
}

FORCE_INLINE void TensorShape::add_dim(int size) {
  shape_.push_back(size);
  if (0 == n_elements_) n_elements_ = 1;
  n_elements_ *= size;
}

} //namespace midend 

#endif

