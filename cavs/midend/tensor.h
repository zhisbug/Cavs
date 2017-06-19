#ifndef CAVS_MIDEND_TENSOR_H_
#define CAVS_MIDEND_TENSOR_H_

#include "cavs/proto/types.pb.h"
#include "cavs/proto/devices.pb.h"
#include "cavs/midend/allocator.h"
#include "cavs/util/logging.h"
#include "cavs/util/macros.h"

#include <vector>
#include <string>
#include <memory>


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
  explicit TensorShape(const std::vector<int>& shape);
  explicit TensorShape(const TensorShape& shape);
  explicit TensorShape(TensorShape&& shape);
  TensorShape& operator =(const TensorShape& b);
  FORCE_INLINE int n_elements() const { return n_elements_; }
  FORCE_INLINE int dims() const { return shape_.size(); }
  FORCE_INLINE int dims(unsigned idx) const {
    CHECK(idx < shape_.size()) << idx << "\t" << shape_.size();
    return shape_.at(idx); 
  }
  FORCE_INLINE TensorShapeDef to_def() const {
    TensorShapeDef def;
    for (int s : shape_)  def.add_dim(s);
    return def;
  }
  void set_dim(int d, int size);
  void add_dim(int size);
  std::string DebugInfo() const;

 private:
  std::vector<int> shape_;
  int n_elements_;
};

class TensorCApi;
class Tensor {
 public:
  Tensor();
  Tensor(const std::string& name, Allocator *a, DataType type, const TensorShape& shape);
  Tensor(const std::string& name, Allocator *a, DataType type, TensorShape&& shape);
  Tensor(const std::string& name, const Tensor& t);
  Tensor(const Tensor& t) { *this = t; }
  Tensor& operator =(const Tensor& t);
  inline DeviceType device_type() const { return buf_->device_type(); }
  inline DataType data_type() const { return type_; }
  inline const std::string& name() const { return name_; }
  inline bool Empty() { return buf_ == nullptr; }
  //for opeators
  inline size_t count() const { return buf_->count(); }
  inline int dims() const { return shape_->dims(); }
  inline int dims(int idx) const { return shape_->dims(idx); }

  void Rebase(Allocator *a, DataType type, const TensorShape& shape);
  void Rebase(Allocator *a, DataType type, TensorShape&& shape);
  void Rebase(Allocator *a, const Tensor& t);
  void Reshape(const TensorShapeDef& shape);
  void Reshape(const std::vector<int>& dims);
  void Reshape(const Tensor& t);
  void SyncWith(const Tensor& t);
  template <typename T>
    T* mutable_data() const { return reinterpret_cast<T*>(buf_->data()); }
  template <typename T>
    const T* data() const { return reinterpret_cast<T*>(buf_->data()); }
  std::string DebugInfo() const;
  template <typename T>
  void DebugNumerical() const;

  friend class TensorCApi;

 private:
  std::shared_ptr<TensorBufferBase> buf_;
  std::shared_ptr<TensorShape> shape_;
  std::string name_;
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
FORCE_INLINE TensorShape::TensorShape(const std::vector<int>& shape) 
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

