#ifndef TENSOR_H_
#define TENSOR_H_

#include "types.pb.h"
#include "allocator.h"

#include <vector>

namespace cavs {

class Tensor{
 public:
  Tensor() {}
  //Tensor(DataType type, const std::vector<int>& shape_);
  //Tensor(DataType type, std::initializer_list<int> shape_);
  //Tensor(Allocator *a, DataType type, std::initializer_list<int> shape);
  Tensor(Allocator *a, DataType type, const TensorShape& shape);
  bool RealAllocate(Allocator *a, DataType type, const TensorShape& shape);
  template<T>
  T* data() const { return reinterpret_cast<T*>(buf_->data()); }
  size_t count() { return buf_->count(); }
  string name() { return name_; }

 private:
  TensorBufferBase* buf_;
  TensorShape shape_;
  string name_;
  //std::vector<int> shape_;
  //DataType data_type;
  //void *data_;
};

class TensorBufferBase {
 public:
  TenosorBufferBase(Allocator* alloc) : alloc_(alloc) {}
  virtual void* data() const = 0;
  //virtual size_t size() const = 0;
  virtual size_t count() const = 0;

 protected:
  Allocator* const alloc_;
};


class TensorShape {
 public:
  TensorShape(vector<int>&& shape);
  void set_dim(int d, int size);
  void add_dim(int size);
  int num_elements() { return num_elements_; }
  int dims() { return shape_.size(); }
 private:
  vector<int> shape_;
  int num_elements_;
};
}
#endif

