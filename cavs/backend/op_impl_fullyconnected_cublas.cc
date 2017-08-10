#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
//#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/stream_event_handle_pool.h"

namespace backend {

using ::midend::Tensor;
using ::midend::Allocator;
using ::midend::GetAllocator;
//using ::midend::DeviceTypeToString;

template <typename T>
class FullyConnectedOpCublas : public OpImpl {
 public:
  explicit FullyConnectedOpCublas(const OpDef& def)
    : OpImpl(def), bias_one_(NULL), bias_length_(0), handle_(NULL) {
    alloc_ = GetAllocator(DeviceTypeToString(GPU));
  }
  void Compute(OpContext* context) override;

 private:
  T* bias_one_;
  int bias_length_;
  Allocator* alloc_;
  cublasHandle_t handle_;
};

template <typename T>
void FullyConnectedOpCublas<T>::Compute(OpContext* context) {
  const Tensor& X = context->Input(0);
  const Tensor& W = context->Input(1);
  const Tensor& B = context->Input(2);
  Tensor* Y = context->Output(0);

  CHECK(X.dims() == 2);
  CHECK(W.dims() == 2);
  CHECK(B.dims() == 2);
  CHECK(Y->dims() == 2);
  int batchN = X.dims(0);
  int K = X.dims(1);
  CHECK(K == W.dims(1));
  int Out = W.dims(0);
  CHECK(Y->dims(0) == batchN);
  CHECK(Y->dims(1) == Out);
  CHECK(B.dims(0) == 1);
  CHECK(B.dims(1) == Out);

  if (batchN != bias_length_) {
    if (bias_one_) {
      alloc_->Deallocate<T>((T*)bias_one_);
    }
    std::vector<float> init;
    init.resize(batchN, 1.f);
    bias_one_ = alloc_->Allocate<T>(batchN);
    checkCudaError(cudaMemcpy(bias_one_, init.data(), batchN*sizeof(float), cudaMemcpyHostToDevice));
    bias_length_ = batchN;
  }

  if (!handle_) {
    if (context->GetStreamID() != -1) {
      handle_ = StreamEventHandlePool::GetCublasHandle(context->GetStreamID());
    }else {
      handle_ = CudaCommon::cublasHandle();
    }
  }

  bool TransX = false;
  bool TransW = true;

  MatMulMatCublasWrapper<T>(handle_, TransX, TransW,
      batchN, Out, K, 1.f, X.data<T>(), W.data<T>(),
      0, Y->mutable_data<T>());
  MatMulMatCublasWrapper<T>(handle_, false, false,
      batchN, Out, 1, 1.f, bias_one_, B.data<T>(),
      1, Y->mutable_data<T>());

  X.DebugNumerical<T>();
  W.DebugNumerical<T>();
  B.DebugNumerical<T>();
  Y->DebugNumerical<T>();
}

template <typename T>
class FullyConnectedGradOpCublas : public OpImpl {
 public:
  explicit FullyConnectedGradOpCublas(const OpDef& def)
    : OpImpl(def), bias_one_(NULL), bias_length_(0), handle_(NULL) {
    alloc_ = GetAllocator(DeviceTypeToString(GPU));
  }
  void Compute(OpContext* context) override;

 private:
  T* bias_one_;
  int bias_length_;
  Allocator* alloc_;
  cublasHandle_t handle_;
};

template <typename T>
void FullyConnectedGradOpCublas<T>::Compute(OpContext* context) {
  const Tensor& dY = context->Input(0);
  const Tensor& X = context->Input(1);
  const Tensor& W = context->Input(2);
  const Tensor& B = context->Input(3);
  Tensor* dW = context->Output(0);
  Tensor* dB = context->Output(1);
  Tensor* dX = context->Output(2);

  CHECK(dY.dims()  == 2);
  CHECK(X.dims()   == 2);
  CHECK(W.dims()   == 2);
  CHECK(B.dims()   == 2);
  CHECK(dW->dims() == 2);
  CHECK(dB->dims() == 2);
  CHECK(dX->dims() == 2);
  for (int i = 0; i < 2; i++) {
    CHECK(X.dims(i) == dX->dims(i));
    CHECK(W.dims(i) == dW->dims(i));
    CHECK(B.dims(i) == dB->dims(i));
  }
  int batchN = X.dims(0);
  int K = X.dims(1);
  CHECK(K == W.dims(1));
  int Out = W.dims(0);
  CHECK(dY.dims(0) == batchN);
  CHECK(dY.dims(1) == Out);
  CHECK(B.dims(0) == 1);
  CHECK(B.dims(1) == Out);

  if (batchN != bias_length_) {
    if (bias_one_) {
      alloc_->Deallocate<T>((T*)bias_one_);
    }
    std::vector<float> init;
    init.resize(batchN, 1.f);
    bias_one_ = alloc_->Allocate<T>(batchN);
    //checkCudaError(cudaGetLastError());
    checkCudaError(cudaMemcpy(bias_one_, init.data(), batchN*sizeof(float), cudaMemcpyHostToDevice));
    bias_length_ = batchN;
  }

  if (!handle_) {
    if (context->GetStreamID() != -1) {
      handle_ = StreamEventHandlePool::GetCublasHandle(context->GetStreamID());
    }else {
      handle_ = CudaCommon::cublasHandle();
    }
  }


  bool Trans_dY = true;
  bool Trans_X  = false;
  MatMulMatCublasWrapper<T>(handle_, Trans_dY, Trans_X,
      Out, K, batchN, 1.f, dY.data<T>(), X.data<T>(),
      0, dW->mutable_data<T>());

  MatMulMatCublasWrapper<T>(handle_, true, false,
      1, Out, batchN, 1.f, bias_one_, dY.data<T>(),
      0, dB->mutable_data<T>());

  MatMulMatCublasWrapper<T>(handle_, false, false,
      batchN, K, Out, 1.f, dY.data<T>(), W.data<T>(),
      0, dX->mutable_data<T>());

  dY.DebugNumerical<T>();
  X.DebugNumerical<T>();
  W.DebugNumerical<T>();
  B.DebugNumerical<T>();
  dW->DebugNumerical<T>();
  dB->DebugNumerical<T>();
  dX->DebugNumerical<T>();
}

REGISTER_OP_IMPL_BUILDER(Key("FullyConnected").Device("GPU"), FullyConnectedOpCublas<float>);
REGISTER_OP_IMPL_BUILDER(Key(GetGradientName("FullyConnected")).Device("GPU"), FullyConnectedGradOpCublas<float>);

} //namespace backend
