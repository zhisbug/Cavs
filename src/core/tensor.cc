#ifndef TENSOR_H_
#define TENSOR_H_

#include "types.pb.h"
#include "allocator.h"

#include <vector>

namespace cavs {

class Tensor{
 public:
  Tensor();
  Tensor(DataType type, const std::vector<int>& shape_);
  Tensor(DataType type, std::initializer_list<int> shape_);
  Tensor(Allocator *a, DataType type, const std::vector<int>& shape_);
 private:
  std::vector<int> shape_;
  DataType data_type;
  void *data_;
};

}
#endif

