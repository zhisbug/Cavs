#ifndef CAVS_BACKEND_OP_IMPL_PLACEHOLDER_H_
#define CAVS_BACKEND_OP_IMPL_PLACEHOLDER_H_

#include "cavs/backend/op_impl.h"
#include "cavs/midend/tensor.h"
#include "cavs/proto/tensor_shape.pb.h"

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

template <typename READFUNCTOR, typename COPYFUNCTOR>//read, copy
class DataOpImpl : public OpImpl {
 public:
  explicit DataOpImpl(const OpDef& def) :
    OpImpl(def), buf_(NULL) {
    batch_ = GetSingleArg<int>(def, "Batch");
    const std::vector<int>& shape = GetListArg<int>(def, "Shape");
    CHECK(!shape.empty());
    CHECK(shape.size() >= 2);
    num_ = shape[0];
    CHECK(batch_ < num_);
    item_size_ = 1;
    for (int i = 1; i < shape.size(); i++)
      item_size_ *= shape[i];
    CHECK(item_size_ > 0);
  }
  ~DataOpImpl() {
    if (buf_)  free(buf_); 
  }

  void Compute(OpContext* context) override {
    Tensor* y = context->Output(0);
    if (!buf_) {
      buf_ = malloc(num_*item_size_);
      READFUNCTOR::Compute(buf_, filename_.c_str(), num_*item_size_);
    }
    int offset = context->GetRound() % (num_/batch_);
    COPYFUNCTOR::Compute(y->mutable_data<char>(), buf_, batch_*item_size_);
  }

 private:
  int batch_;
  int num_;
  int item_size_;
  std::string filename_;
  void* buf_;
};

} //namespace cavs

#endif
