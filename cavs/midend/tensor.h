#ifndef CAVS_MIDEND_TENSOR_H_
#define CAVS_MIDEND_TENSOR_H_

#include "cavs/midend/types.pb.h"
#include "cavs/midend/device.pb.h"
#include "cavs/midend/macros.h"
#include "cavs/midend/allocator.h"
#include "cavs/util/logging.h"

#include <vector>
#include <string>
#include <memory>

using std::vector;
using std::string;

namespace cavs {

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

class TensorShape {
 public:
  TensorShape() : n_elements_(0) {}
  typedef OpDef::AttrType::ListValue ListValue;
  explicit TensorShape(const ListValue& shape);
  explicit TensorShape(std::initializer_list<int> shape);
  explicit TensorShape(const TensorShape& shape);
  explicit TensorShape(TensorShape&& shape);
  TensorShape& operator =(const TensorShape& b);
  FORCE_INLINE int n_elements() const { return n_elements_; }
  FORCE_INLINE int dims() const { return shape_.size(); }
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
  //Tensor(const string& name) : name_(name) {}
  Tensor(const string& name, Allocator *a, DataType type, const TensorShape& shape);
  Tensor(const string& name, Allocator *a, DataType type, TensorShape&& shape);
  FORCE_INLINE DeviceType device_type() const { return buf_->type(); }
  FORCE_INLINE const string& name() const { return name_; }
  FORCE_INLINE Tensor& operator =(const Tensor& tensor) {
    buf_ = tensor.buf_;
    shape_ = tensor.shape_;
    name_ = tensor.name_;
    return *this;
  }

  void Reshape(Allocator *a, DataType type, const TensorShape& shape);
  template <typename T>
    T* mutable_data() const { return reinterpret_cast<T*>(buf_->data()); }
  template <typename T>
    const T* data() const { return reinterpret_cast<T*>(buf_->data()); }

  friend class TensorCApi;
  //FORCE_INLINE size_t count() const { return buf_->count(); }

 private:
  std::shared_ptr<TensorBufferBase> buf_;
  std::shared_ptr<TensorShape> shape_;
  string name_;
};

FORCE_INLINE TensorShape::TensorShape(const ListValue& shape) {
  CHECK(shape.i_size() > 0);
  shape_.resize(shape.i_size());
  n_elements_ = 1;    
  for (int idx = 0; idx < shape.i_size(); idx++) {
    shape_[idx] = shape.i(idx);
    n_elements_ *= shape.i(idx); 
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
  shape_[d] = size;
}

FORCE_INLINE void TensorShape::add_dim(int size) {
  shape_[dims()] = size;
  n_elements_ *= size;
}

} //namespace cavs

#endif

