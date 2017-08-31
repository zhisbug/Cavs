#ifndef CAVS_BACKEND_OPS_IMPL_ELEMENTWISE_COMMON_H_
#define CAVS_BACKEND_OPS_IMPL_ELEMENTWISE_COMMON_H_

#include "cavs/backend/op_impl.h"
#include "cavs/midend/allocator.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/stream_event_handle_pool.h"

using ::midend::Tensor;

namespace backend {

template <typename FUNCTOR, typename T>//mathop, dtype
class UnaryOp : public OpImpl {
 public:
  explicit UnaryOp(const OpDef& def) :
    OpImpl(def), stream_(cudaStreamDefault) {}

  void Compute(OpContext* context) override {
    const Tensor& inp = context->Input(0);
    inp.DebugNumerical<T>();
    Tensor* out = context->Output(0);

    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[Unary] Assign new stream with ID " << context->GetStreamID();
    }

    FUNCTOR::Compute(out->mutable_data<T>(), out->count(), 
        inp.data<T>(), inp.count(), stream_);
    out->DebugNumerical<T>();
  }

 private:
  cudaStream_t stream_;
};

template <typename FUNCTOR, typename T>
class BinaryOp : public OpImpl {
 public:
  explicit BinaryOp(const OpDef& def) :
    OpImpl(def), stream_(cudaStreamDefault) {}

  void Compute(OpContext* context) override {
    const Tensor& inp0 = context->Input(0);
    const Tensor& inp1 = context->Input(1);
    inp0.DebugNumerical<T>();
    inp1.DebugNumerical<T>();
    Tensor* out = context->Output(0);

    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[Binary] Assign new stream with ID " << context->GetStreamID();
    }

    FUNCTOR::Compute(out->mutable_data<T>(), out->count(), 
        inp0.data<T>(), inp0.count(), inp1.data<T>(), inp1.count(), stream_);
    out->DebugNumerical<T>();
  }

 private:
  cudaStream_t stream_;
};

template <typename FUNCTOR, typename T>
class PartialAccumulateBinaryOp : public OpImpl {
 public:
  explicit PartialAccumulateBinaryOp(const OpDef& def) : OpImpl(def),
      split_(-1), index_(-1), offset_(-1), stride_(-1), stream_(cudaStreamDefault) {
    if (GetSingleArg(def, "Split", 0) != 0) {
      //dynamic slicing
      split_ = GetSingleArg<int>(def, "Split"); 
      index_ = GetSingleArg<int>(def, "Index"); 
      CHECK(split_ > 0);
      CHECK(index_ >= 0);
    }else {
      //static slicing
      offset_ = GetSingleArg<int>(def, "Offset");
      stride_ = GetSingleArg<int>(def, "Stride");
      CHECK(offset_ >= 0);
      CHECK(stride_ > 0);
    }
  }
  void Compute(OpContext* context) override {
    //The partialadd(+=) operator behaves like a binary operation
    //But it actually has one input, and the output is both input and output
    const Tensor& inp = context->Input(0);
    Tensor* out = context->Output(0);
    out->DebugNumerical<T>();

    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[PartialAccumulate] Assign new stream with ID " << context->GetStreamID();
    }

    if (inp.IsDynamicShape()) {
      CHECK(out->IsDynamicShape());
      CHECK(!inp.IsFullShape());
      CHECK(!out->IsFullShape());
      CHECK(inp.dims() == 2);
      CHECK(out->dims() == 2);
      CHECK(out->dims(0) == inp.dims(0));
      CHECK((out->count()/out->dims(0)) % split_ == 0);
      CHECK(offset_ < 0);

      int dyn_dim = out->dims(0);
      int out_stride = out->count() / dyn_dim;
      int inp_stride = out_stride / split_;
      int inp_offset = inp_stride * index_;
      CHECK(inp_stride = inp.count()/dyn_dim);
      CHECK(out_stride % inp_stride == 0);
      //for (int i = 0; i < dyn_dim; i++) {
        //int out_offset = inp_offset + i*out_stride;
        //FUNCTOR::Compute(out->mutable_data<T>() + out_offset, inp_stride,
                         //out->mutable_data<T>() + out_offset, inp_stride,
                         //inp.data<T>() + i*inp_stride, inp_stride, stream_);
      //}
      FUNCTOR::Compute(out->mutable_data<T>() + inp_offset, out_stride, 
                       out->data<T>() + inp_offset, out_stride,
                       inp.data<T>(), inp_stride,
                       dyn_dim, inp_stride, stream_);
    }else {
      //inp is the small tensor and out is the big one
      CHECK(!out->IsDynamicShape());
      CHECK(out->dims(0) == 1);
      CHECK(inp.dims(0) == 1);
      if (split_ > 0) {
        //it means the dynamic slicing
        CHECK(out->count() % split_ == 0);
        stride_ = out->count() / split_;
        offset_ = stride_ * index_;
      }
      CHECK(inp.count() == stride_);
      //FUNCTOR::Compute(out->mutable_data<T>() + offset_, stride_,
                       //out->mutable_data<T>() + offset_, stride_,
                       //inp.data<T>(), inp.count(), stream_);
      FUNCTOR::Compute(out->mutable_data<T>() + offset_, out->count(), 
                       out->data<T>() + offset_, out->count(),
                       inp.data<T>(), inp.count(),
                       1, inp.count(), stream_);
    }
    inp.DebugNumerical<T>();
    out->DebugNumerical<T>();
  }

 private:
  int offset_;
  int stride_;
  int split_;
  int index_;
  cudaStream_t stream_;
};

} //namespace backend

#endif
