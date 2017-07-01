#ifndef CAVS_BACKEND_OP_IMPL_PLACEHOLDER_H_
#define CAVS_BACKEND_OP_IMPL_PLACEHOLDER_H_

#include "cavs/backend/op_impl.h"
#include "cavs/midend/tensor.h"
#include "cavs/proto/tensor_shape.pb.h"

#include <string>
#include <mpi.h>

namespace backend {

using ::midend::OpContext;
using ::midend::Tensor;

class PlaceholderOpImpl : public OpImpl {
 public:
  explicit PlaceholderOpImpl(const OpDef& def) : OpImpl(def) {}

  void Compute(OpContext* context) override {
    //do nothing now
  }
};

template <typename READFUNCTOR, typename COPYFUNCTOR, typename T, bool MPIEnable>//read, copy
class DataOpImpl : public OpImpl {
 public:
  explicit DataOpImpl(const OpDef& def) :
    OpImpl(def), buf_(NULL), curr_idx_(-1) {
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
    if (MPIEnable) {
      int size;
      MPI_Comm_size(MPI_COMM_WORLD, &size); 
      num_ /= size;
    }
  }
  ~DataOpImpl() {
    if (buf_)  free(buf_); 
  }

  void Compute(OpContext* context) override {
    if (!buf_) {
      buf_ = (T*)malloc(num_*item_size_*sizeof(T));
      READFUNCTOR::Compute(buf_, filename_.c_str(), num_*item_size_*sizeof(T));
    }
    int next_idx = context->GetRound() % (num_/batch_);
    if (next_idx != curr_idx_) {
      Tensor* out = context->Output(0);
      CHECK(out->count() == batch_*item_size_);
      CHECK(next_idx >= 0 && next_idx < num_/batch_);
      COPYFUNCTOR::Compute(out->mutable_data<T>(), buf_+next_idx*batch_*item_size_, batch_*item_size_*sizeof(T));
      curr_idx_ = next_idx;
    }
  }

 private:
  int curr_idx_;
  int batch_;
  int num_;
  int item_size_;
  std::string filename_;
  T* buf_;
};

} //namespace cavs

#endif
