#ifndef CAVS_BACKEND_OP_IMPL_PLACEHOLDER_H_
#define CAVS_BACKEND_OP_IMPL_PLACEHOLDER_H_

#include "cavs/backend/op_impl.h"
#include "cavs/midend/tensor.h"
#include "cavs/proto/tensor_shape.pb.h"

#include <string>

namespace backend {

using ::midend::OpContext;
using ::midend::Tensor;

template <typename COPYFUNCTOR>//copy
class PlaceholderOpImpl : public OpImpl {
 public:
  explicit PlaceholderOpImpl(const OpDef& def) : OpImpl(def) {}

  void Compute(OpContext* context) override {
    //do nothing now
  }
};

template <typename READFUNCTOR, typename COPYFUNCTOR, typename T>//read, copy
class DataOpImpl : public OpImpl {
 public:
  explicit DataOpImpl(const OpDef& def) :
    OpImpl(def), buf_(NULL) {
    batch_ = GetSingleArg<int>(def, "Batch");
    const std::vector<int>& shape = GetListArg<int>(def, "Shape");
    CHECK(!shape.empty());
    CHECK(shape.size() >= 2);
    num_ = shape[0];
    CHECK(batch_ <= num_) << def.DebugString();
    item_size_ = 1;
    for (int i = 1; i < shape.size(); i++)
      item_size_ *= shape[i];
    CHECK(item_size_ > 0);
    filename_ = GetSingleArg<std::string>(def, "filename");
    CHECK(filename_.length() > 0);
  }
  ~DataOpImpl() {
    if (buf_)  free(buf_); 
  }

  void Compute(OpContext* context) override {
    Tensor* y = context->Output(0);
    if (!buf_) {
      buf_ = (T*)malloc(num_*item_size_*sizeof(T));
      READFUNCTOR::Compute(buf_, filename_.c_str(), num_*item_size_*sizeof(T));
    }
    int offset = context->GetRound() % (num_/batch_);
    COPYFUNCTOR::Compute(y->mutable_data<T>(), buf_+offset, batch_*item_size_*sizeof(T));
  }

 private:
  int batch_;
  int num_;
  int item_size_;
  std::string filename_;
  T* buf_;
};

} //namespace cavs

#endif
