#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/cudnn_types.h"

using std::vector;

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
  CHECK(x.dims() == y->dims());
  CHECK(x.dims() < 5);
  CHECK(x.dims() > 1);

  //I don't know why cudnn has this bug(in NdTensor support)...
  int XN, YN, XC, YC, XH, YH, XW, YW;
  XN = YN = x.dims(0);
  XC = YC = x.dims(1);
  XH = YH = x.dims() > 2? x.dims(2) : 1;
  XW = YW = x.dims() > 3? x.dims(3) : 1;

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
  /*x.DebugNumerical<T>();*/
  /*y->DebugNumerical<T>();*/
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

  CHECK(x.dims() == y.dims());
  CHECK(x.dims() < 5);
  CHECK(x.dims() > 1);
  //I don't know why cudnn has this bug(in NdTensor support)...
  int XN, YN, XC, YC, XH, YH, XW, YW;
  XN = YN = x.dims(0);
  XC = YC = x.dims(1);
  XH = YH = x.dims() > 2? x.dims(2) : 1;
  XW = YW = x.dims() > 3? x.dims(3) : 1;

  float alpha = 1.f, beta = 0.f;
  checkCUDNNError(cudnnSetTensor4dDescriptor(x_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  XN, XC, XH, XW));
  checkCUDNNError(cudnnSetTensor4dDescriptor(y_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  YN, YC, YH, YW));
  checkCUDNNError(cudnnActivationBackward(CudaCommon::cudnnHandle(),
                  activation_desc_,
                  &alpha, y_desc_, y.data<T>(),
                  y_desc_, dy.data<T>(),
                  x_desc_, x.data<T>(),
                  &beta,
                  x_desc_, dx->mutable_data<T>()));
  /*dy.DebugNumerical<T>();*/
  /*y.DebugNumerical<T>();*/
  /*x.DebugNumerical<T>();*/
  /*dx->DebugNumerical<T>();*/
}

REGISTER_OP_IMPL_BUILDER(Key("Relu").Device("GPU"), ActivationOpCudnn<float>);
REGISTER_OP_IMPL_BUILDER(Key(GetGradientName("Relu")).Device("GPU"), ActivationOpCudnnGrad<float>);

} //namespace backend
