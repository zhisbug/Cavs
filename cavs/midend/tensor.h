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
  virtual void* Resize(int size) = 0;

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
  FORCE_INLINE int dim() const { return shape_.size(); }
  FORCE_INLINE int dim(unsigned idx) const {
    CHECK(idx < shape_.size()) << idx << "\t" << shape_.size();
    return shape_.at(idx); 
  }
  FORCE_INLINE TensorShapeDef to_def() const {
    TensorShapeDef def;
    for (int s : shape_)  def.add_dim(s);
    return def;
  }
  void SetDim(int d, int size);
  void AddDim(int size);
  std::string debug_info() const;

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
  inline DataType data_type()     const { return type_; }
  inline std::string name()       const { return name_; }
  inline bool empty()             const { return buf_ == nullptr; }
  inline bool IsDynamicSize()     const { return dynamic_size_; }
  //for opeators
  inline size_t count()    const { return shape_->n_elements(); }
  inline int dims()        const { return shape_->dim(); }
  inline int dims(int idx) const { return shape_->dim(idx); }

  //allocate a new buffer
  void Rebase(Allocator *a, DataType type, const TensorShape& shape);
  void Rebase(Allocator *a, DataType type, TensorShape&& shape);
  void Rebase(Allocator *a, const Tensor& t);
  //reuse pre-allocated buffer and only change the shape
  void Reshape(const TensorShapeDef& shape);
  void Reshape(const std::vector<int>& dims);
  void Reshape(const Tensor& t);
  void ScaleShape(int dyn_dim);
  template <typename T>
    T* mutable_data() const { return reinterpret_cast<T*>(buf_->data()); }
  template <typename T>
    const T* data() const { return reinterpret_cast<T*>(buf_->data()); }

  inline bool IsShareBuf(const Tensor& t) const { return buf_ && t.buf_ && buf_ == t.buf_; }
  void SyncWith(const Tensor& t);

  std::string debug_info() const;
  template <typename T>
  void DebugNumerical() const;

  friend class TensorCApi;

 private:
  std::shared_ptr<TensorBufferBase> buf_;
  std::shared_ptr<TensorShape> shape_;
  std::string name_;
  DataType type_;
  //if a tensor is dynamic at first, it is dynamic at last
  bool dynamic_size_;
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

FORCE_INLINE void TensorShape::SetDim(int d, int size) {
  CHECK(dim() > d);
  n_elements_ = n_elements_/shape_[d]*size;
  shape_[d] = size;
}

FORCE_INLINE void TensorShape::AddDim(int size) {
  shape_.push_back(size);
  if (0 == n_elements_) n_elements_ = 1;
  n_elements_ *= size;
}

} //namespace midend 

#endif

