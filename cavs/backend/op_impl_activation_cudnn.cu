#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/cudnn_types.h"

namespace backend {

using ::midend::Tensor;

class ActivationOpCudnnBase : public OpImpl {
 public:
  explicit ActivationOpCudnnBase(const OpDef& def);
  ~ActivationOpCudnnBase();

 protected:
  cudnnTensorDescriptor_t x_desc_, y_desc_;
  cudnnActivationDescriptor_t activation_desc_;
};

ActivationOpCudnnBase::ActivationOpCudnnBase(const OpDef& def)
    : OpImpl(def) {
  checkCUDNNError(cudnnCreateTensorDescriptor(&x_desc_));    
  checkCUDNNError(cudnnCreateTensorDescriptor(&y_desc_));    
  checkCUDNNError(cudnnCreateActivationDescriptor(&activation_desc_));    
  //Currently, the arguments are hard coded.
  checkCUDNNError(cudnnSetActivationDescriptor(activation_desc_,
        CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));    
}

ActivationOpCudnnBase::~ActivationOpCudnnBase() {
  checkCUDNNError(cudnnDestroyTensorDescriptor(x_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(y_desc_));
  checkCUDNNError(cudnnDestroyActivationDescriptor(activation_desc_));
}

template <typename T>
class ActivationOpCudnn : public ActivationOpCudnnBase {
 public:
  explicit ActivationOpCudnn(const OpDef& def) 
      : ActivationOpCudnnBase(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
void ActivationOpCudnn<T>::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  Tensor* y = context->Output(0);
  int XN = x.dims(0);
  int XC = x.dims(1);
  int XH = x.dims(2);
  int XW = x.dims(3);
  int YN = y->dims(0);
  int YC = y->dims(1);
  int YH = y->dims(2);
  int YW = y->dims(3);

  CHECK(XN == YN);
  CHECK(XC == YC);
  CHECK(XH == YH);
  CHECK(XW == YW);

  float alpha = 1.f, beta = 0.f;
  checkCUDNNError(cudnnSetTensor4dDescriptor(x_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  XN, XC, XH, XW));
  checkCUDNNError(cudnnSetTensor4dDescriptor(y_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  YN, YC, YH, YW));
  checkCUDNNError(cudnnActivationForward(CudaCommon::cudnnHandle(),
                  activation_desc_,
                  &alpha, x_desc_, x.data<T>(),
                  &beta, y_desc_, y->mutable_data<T>()));
}

template <typename T>
class ActivationOpCudnnGrad : public ActivationOpCudnnBase {
 public:
  explicit ActivationOpCudnnGrad(const OpDef& def) 
      : ActivationOpCudnnBase(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
void ActivationOpCudnnGrad<T>::Compute(OpContext* context) {
  const Tensor& dy = context->Input(0);
  const Tensor& y = context->Input(1);
  const Tensor& x = context->Input(2);
  Tensor* dx = context->Output(0);

  CHECK(dy.dims(0) == y.dims(0) == x.dims(0) == dx->dims(0));
  CHECK(dy.dims(1) == y.dims(1) == x.dims(1) == dx->dims(1));
  CHECK(dy.dims(2) == y.dims(2) == x.dims(2) == dx->dims(2));
  CHECK(dy.dims(3) == y.dims(3) == x.dims(3) == dx->dims(3));

  float alpha = 1.f, beta = 0.f;
  checkCUDNNError(cudnnSetTensor4dDescriptor(x_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  x.dims(0), x.dims(1), x.dims(2), x.dims(3)));
  checkCUDNNError(cudnnSetTensor4dDescriptor(y_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  y.dims(0), y.dims(1), y.dims(2), y.dims(3)));
  checkCUDNNError(cudnnActivationBackward(CudaCommon::cudnnHandle(),
                  activation_desc_,
                  &alpha, y_desc_, y.data<T>(),
                  y_desc_, dy.data<T>(),
                  x_desc_, x.data<T>(),
                  &beta,
                  x_desc_, dx->mutable_data<T>()));
}

} //namespace backend
