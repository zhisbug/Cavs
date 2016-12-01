#ifndef CAVS_MIDEND_TENSOR_H_
#define CAVS_MIDEND_TENSOR_H_

#include "cavs/midend/types.pb.h"
#include "cavs/midend/macros.h"
#include "cavs/midend/allocator.h"

#include <vector>
#include <string>
#include <memory>

using std::vector;
using std::string;

namespace cavs {

class TensorBufferBase {
 public:
  TensorBufferBase(Allocator* alloc) : alloc_(alloc) {}
  virtual ~TensorBufferBase() {}
  virtual void* data() const = 0;
  //virtual size_t size() const = 0;
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
  TensorShape& operator = (const TensorShape& b);
  void set_dim(int d, int size);
  void add_dim(int size);
  int n_elements() const { return n_elements_; }
  int dims() const { return shape_.size(); }
 private:
  //TensorShape(vector<int>& shape);
  vector<int> shape_;
  int n_elements_;
};

FORCE_INLINE TensorShape& TensorShape::operator = (const TensorShape& b) {
  n_elements_ = b.n_elements_;
  shape_ = b.shape_;
  return *this;
}

class Tensor {
 public:
  Tensor(const string& name) : name_(name) {}
  //Tensor(DataType type, const std::vector<int>& shape_);
  //Tensor(DataType type, std::initializer_list<int> shape_);
  //Tensor(Allocator *a, DataType type, std::initializer_list<int> shape);
  Tensor(const string& name, Allocator *a, DataType type, const TensorShape& shape);
  void Reshape(Allocator *a, DataType type, const TensorShape& shape);
  template <typename T>
    T* mutable_data() const { return reinterpret_cast<T*>(buf_->data()); }
  template <typename T>
    const T* data() const { return reinterpret_cast<T*>(buf_->data()); }
  size_t count() const { return buf_->count(); }
  string name() const { return name_; }
  const TensorShape& shape() const { return shape_; }

 private:
  std::unique_ptr<TensorBufferBase> buf_;
  TensorShape shape_;
  const string name_;
};

} //namespace cavs

#endif

