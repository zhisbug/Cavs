#include "cavs/backend/op_impl.h"
#include "cavs/midend/tensor.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/stream_event_handle_pool.h"

using std::vector;
using std::string;
using ::midend::Tensor;

namespace backend {

template <typename T>
class SliceOpImpl : public OpImpl {
 public:
  explicit SliceOpImpl(const OpDef& def) :
    OpImpl(def), split_(-1), index_(-1), offset_(-1), stride_(-1),
    stream_(cudaStreamDefault) {
    CHECK(!GetSingleArg<bool>(op_def_, "ShareMemory", false));
    //currently, we only support axis equals 0
    CHECK((axis_ = GetSingleArg<int>(op_def_, "Axis", 0)) == 0);
    if (GetSingleArg(def, "Split", 0) != 0) {
      split_ = GetSingleArg<int>(def, "Split"); 
      index_ = GetSingleArg<int>(def, "Index"); 
      CHECK(split_ > 0);
      CHECK(index_ >= 0);
    }else {
      offset_ = GetSingleArg<int>(def, "Offset");
      stride_ = GetSingleArg<int>(def, "Stride");
      CHECK(offset_ >= 0);
      CHECK(stride_ > 0);
    }
  }

  void Compute(OpContext* context) override {
    const Tensor& x = context->Input(0);
    Tensor* y = context->Output(0);

    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[Slice] Assign new stream with ID " << context->GetStreamID();
    }

    CHECK(axis_ == 0);
    if (x.IsDynamicShape()) {
      CHECK(y->IsDynamicShape());
      CHECK(y->dims(0) == x.dims(0));
      CHECK(offset_ < 0) << "dynamic support for static slice is not available now";
      CHECK(split_ > 0);
      CHECK(index_ >= 0);
      CHECK((x.count() / x.dims(0)) % split_ == 0);
      CHECK(x.dims() == 2) << "In fact, we support the dimension larger than 2";
      CHECK(y->dims() == 2);
      int dyn_dim = x.dims(0);
      int x_stride = x.count() / dyn_dim;
      int y_stride = x_stride / split_;
      int x_offset = y_stride * index_;
      CHECK(dyn_dim * y_stride == y->count());
      for (int i = 0; i < dyn_dim; i++) {
        checkCudaError(cudaMemcpyAsync(y->mutable_data<T>() + i*y_stride,
                                       x.data<T>() + x_offset + i*x_stride,
                                       y_stride * sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream_));
      }
    }else {
      //static slicing
      CHECK(!y->IsDynamicShape());
      if (offset_ < 0) {
        CHECK(x.count()% split_ == 0);
        stride_ = x.count() / split_;
        offset_ = x.count() / split_ * index_;
      }
      CHECK(stride_ == y->count());
      checkCudaError(cudaMemcpyAsync(y->mutable_data<T>(),
                                     x.data<T>()+offset_,
                                     stride_*sizeof(T),
                                     cudaMemcpyDeviceToDevice, stream_));
    }
  }

 private:
  int offset_;
  int stride_;
  int split_;
  int index_;
  int axis_;
  cudaStream_t stream_;
};

template <typename T>
class ConcatOpImpl : public OpImpl {
 public:
  explicit ConcatOpImpl(const OpDef& def) :
    OpImpl(def), stream_(cudaStreamDefault) {
    CHECK((axis_ = GetSingleArg<int>(op_def_, "Axis", 0)) == 0);
  }
  void Compute(OpContext* context) override {
    Tensor* out = context->Output(0);
    CHECK(out->count() > 0);

    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[Concat] Assign new stream with ID " << context->GetStreamID();
    }

    int copied_count = 0;
    for (int i = 0; i < context->InputSize(); i++) {
      const Tensor& inp = context->Input(i);
      CHECK(inp.count() > 0);
      CHECK(copied_count + inp.count() <= out->count());
      if (out->IsDynamicShape()) {
        CHECK(inp.IsDynamicShape());
        CHECK(out->dims() == 2);
        CHECK(inp.dims() == 2);
        CHECK(inp.dims(0) == out->dims(0));
        int dyn_dim = out->dims(0);
        int out_stride = out->count() / dyn_dim;
        int inp_stride = inp.count() / dyn_dim;
        for (int j = 0; j < dyn_dim; j++) {
          int out_offset = copied_count / dyn_dim;
          checkCudaError(cudaMemcpyAsync(out->mutable_data<T>() + out_offset + j*out_stride,
                                         inp.data<T>() + j*inp_stride,
                                         inp_stride * sizeof(T),
                                         cudaMemcpyDeviceToDevice, stream_));
        }
      }else {
        CHECK(!inp.IsDynamicShape());
        checkCudaError(cudaMemcpyAsync(out->mutable_data<T>()+copied_count,
                                       inp.data<T>(),
                                       inp.count()*sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream_));
      }
      copied_count += inp.count();
      inp.DebugNumerical<T>();
    }

    CHECK(out->count() == copied_count);
    out->DebugNumerical<T>();
  }

 private:
  int axis_;
  cudaStream_t stream_;
};

template <typename T>
class SliceAllOpImpl : public OpImpl {
 public:
  explicit SliceAllOpImpl(const OpDef& def) :
    OpImpl(def), stream_(cudaStreamDefault) {
    CHECK((axis_ = GetSingleArg<int>(op_def_, "Axis", 0)) == 0);
  }

  void Compute(OpContext* context) override {
    CHECK(context->InputSize() == context->OutputSize()+1);
    const Tensor& input = context->Input(0);
    input.DebugNumerical<T>();
    CHECK(input.count() > 0);

    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[SliceAll] Assign new stream with ID " << context->GetStreamID();
    }

    int copied_count = 0;
    for (int i = 0; i < context->OutputSize(); i++) {
      const Tensor& inp_check = context->Input(i+1);
      Tensor* out = context->Output(i);
      CHECK(inp_check.count() == out->count());
      CHECK(copied_count + out->count() <= input.count());

      if (input.IsDynamicShape()) {
        CHECK(out->IsDynamicShape());
        CHECK(!input.IsFullShape());
        CHECK(!out->IsFullShape());
        CHECK(input.dims() == 2);
        CHECK(out->dims() == 2);
        CHECK(out->dims(0) == input.dims(0));
        int dyn_dim = input.dims(0);
        int inp_stride = input.count() / dyn_dim;
        int out_stride = out->count() / dyn_dim;
        for (int j = 0; j < dyn_dim; j++) {
          int input_offset = copied_count / dyn_dim;
          checkCudaError(cudaMemcpyAsync(out->mutable_data<T>() + j*out_stride,
                                         input.data<T>() + input_offset + j*inp_stride,
                                         out_stride*sizeof(T),
                                         cudaMemcpyDeviceToDevice, stream_));
        }
      }else {
        CHECK(!out->IsDynamicShape());
        checkCudaError(cudaMemcpyAsync(out->mutable_data<T>(),
                                       input.data<T>()+copied_count,
                                       out->count()*sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream_));
      }

      copied_count += out->count();
      out->DebugNumerical<T>();
    }
    CHECK(input.count() == copied_count);
  }

 private:
  int axis_;
  cudaStream_t stream_;
};

class MirrorOpImpl : public OpImpl {
 public:
  explicit MirrorOpImpl(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    //do nothing 
  }
};

REGISTER_OP_IMPL_BUILDER(Key("Slice").Device("GPU"),    SliceOpImpl<float>);
REGISTER_OP_IMPL_BUILDER(Key("Concat").Device("GPU"),   ConcatOpImpl<float>);
REGISTER_OP_IMPL_BUILDER(Key("SliceAll").Device("GPU"), SliceAllOpImpl<float>);
REGISTER_OP_IMPL_BUILDER(Key("Mirror").Device("GPU"), MirrorOpImpl);

} //namespace backend
