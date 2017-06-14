#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
#include "cavs/backend/functor_elementwise.h"
#include "cavs/backend/op_impl_elementwise.cuh"
#include "cavs/backend/functor_reduction.cuh"

namespace backend {

using ::midend::Tensor;

//absolute value mean
template <typename T>
class AmeanOpCublas : public OpImpl {
 public:
  explicit AmeanOpCublas(const OpDef& def)
    : OpImpl(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
void AmeanOpCublas<T>::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  Tensor* y = context->Output(0);
  CHECK(1 == y->count());
  int N = x.count();
  CHECK(N > 0);

  x.DebugNumerical<T>();
  AsumCublasWrapper<T>(
      N, x.data<T>(), y->mutable_data<T>());
  CUDABinaryConstScalarFunctor<math::Div<T>, T>:: Compute(
      y->mutable_data<T>(), y->count(), 
      y->mutable_data<T>(), y->count(), 
      N);
  y->DebugNumerical<T>();
}

//absolute value argmax
template <typename T>
class ArgmaxOpCublasOrCuda: public OpImpl {
 public:
  explicit ArgmaxOpCublasOrCuda(const OpDef& def)
    : OpImpl(def) {
    axis_ = GetSingleArg<int>(op_def_, "Axis");
  }
  void Compute(OpContext* context) override;

 private:
  int axis_;
};

template <typename T>
void ArgmaxOpCublasOrCuda<T>::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  Tensor* y = context->Output(0);
  if (axis_ == 0) {
    CHECK(1 == y->count()) << op_def_.DebugString();
    int N = x.count();
    CHECK(N > 0);
    ArgmaxCublasWrapper<T>(N, x.data<T>(), y->mutable_data<int>());
  }else {
    CHECK(axis_ >= 1);
    CHECK(x.dims() > axis_);
    CHECK(y->dims() == axis_+1);
    CHECK(y->dims(axis_) == 1);
    int BATCH = 1;
    for (int i = 0; i < axis_; i++) {
      CHECK(x.dims(i) == y->dims(i));
      BATCH *= x.dims(i);
    }
    int N = x.count()/BATCH;
    T* out_value = NULL;
    BatchedArgmax(out_value, y->mutable_data<int>(), x.data<T>(), N, BATCH);
  }
  CUDAUnaryFunctor<math::Cast<T, int>, T, int>::Compute(y->mutable_data<T>(),
      y->count(), y->data<int>(), y->count());

  /*x.DebugNumerical<T>();*/
  /*y->DebugNumerical<T>();*/
}


REGISTER_OP_IMPL_BUILDER(Key("Reduce_mean").Device("GPU"), AmeanOpCublas<float>);
REGISTER_OP_IMPL_BUILDER(Key("Argmax").Device("GPU"), ArgmaxOpCublasOrCuda<float>);

} //namespace backend

