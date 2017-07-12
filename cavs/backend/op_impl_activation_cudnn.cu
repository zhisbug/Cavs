#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
/*#include "cavs/midend/devices.h"*/
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/cudnn_types.h"

using std::vector;

namespace backend {

using ::midend::Tensor;

template <cudnnActivationMode_t mode>
class ActivationOpCudnnBase : public OpImpl {
 public:
  explicit ActivationOpCudnnBase(const OpDef& def);
  ~ActivationOpCudnnBase();

 protected:
  cudnnTensorDescriptor_t x_desc_, y_desc_;
  cudnnActivationDescriptor_t activation_desc_;
};

template <cudnnActivationMode_t mode>
ActivationOpCudnnBase<mode>::ActivationOpCudnnBase(const OpDef& def)
    : OpImpl(def) {
  checkCUDNNError(cudnnCreateTensorDescriptor(&x_desc_));    
  checkCUDNNError(cudnnCreateTensorDescriptor(&y_desc_));    
  checkCUDNNError(cudnnCreateActivationDescriptor(&activation_desc_));    
  /*//Currently, the arguments are hard coded.*/
  /*checkCUDNNError(cudnnSetActivationDescriptor(activation_desc_,*/
        /*CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));    */
  checkCUDNNError(cudnnSetActivationDescriptor(activation_desc_,
        mode, CUDNN_NOT_PROPAGATE_NAN, 0));    
}

template <cudnnActivationMode_t mode>
ActivationOpCudnnBase<mode>::~ActivationOpCudnnBase() {
  checkCUDNNError(cudnnDestroyTensorDescriptor(x_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(y_desc_));
  checkCUDNNError(cudnnDestroyActivationDescriptor(activation_desc_));
}

template <typename T, cudnnActivationMode_t mode>
class ActivationOpCudnn : public ActivationOpCudnnBase<mode> {
 public:
  explicit ActivationOpCudnn(const OpDef& def) 
      : ActivationOpCudnnBase<mode>(def) {}
  void Compute(OpContext* context) override;

  using ActivationOpCudnnBase<mode>::x_desc_;
  using ActivationOpCudnnBase<mode>::y_desc_;
  using ActivationOpCudnnBase<mode>::activation_desc_;
};

template <typename T, cudnnActivationMode_t mode>
void ActivationOpCudnn<T, mode>::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  Tensor* y = context->Output(0);
  CHECK(x.dims() == y->dims());
  CHECK(x.dims() < 5) << x.dims();
  CHECK(x.dims() > 0) << x.dims();

  //I don't know why cudnn has this bug(in NdTensor support)...
  //it can be one-dimension
  //and therefore we loose the constraint in the development of tree-lstm
  int XN, YN, XC, YC, XH, YH, XW, YW;
  XN = YN = x.dims(0);
  XC = YC = x.dims() > 1? x.dims(1) : 1;
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

template <typename T, cudnnActivationMode_t mode>
class ActivationOpCudnnGrad : public ActivationOpCudnnBase<mode> {
 public:
  explicit ActivationOpCudnnGrad(const OpDef& def) 
      : ActivationOpCudnnBase<mode>(def) {}
  void Compute(OpContext* context) override;

  using ActivationOpCudnnBase<mode>::x_desc_;
  using ActivationOpCudnnBase<mode>::y_desc_;
  using ActivationOpCudnnBase<mode>::activation_desc_;
};

template <typename T, cudnnActivationMode_t mode>
void ActivationOpCudnnGrad<T, mode>::Compute(OpContext* context) {
  const Tensor& dy = context->Input(0);
  const Tensor& y = context->Input(1);
  const Tensor& x = context->Input(2);
  Tensor* dx = context->Output(0);

  CHECK(x.dims() == y.dims());
  CHECK(x.dims() < 5) << x.dims();
  CHECK(x.dims() > 0) << x.dims();

  //I don't know why cudnn has this bug(in NdTensor support)...
  //it can be one-dimension
  //and therefore we loose the constraint in the development of tree-lstm
  int XN, YN, XC, YC, XH, YH, XW, YW;
  XN = YN = x.dims(0);
  XC = YC = x.dims() > 1? x.dims(1) : 1;
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

REGISTER_OP_IMPL_BUILDER(Key("Relu").Device("GPU"),    ActivationOpCudnn<float, CUDNN_ACTIVATION_RELU>);
REGISTER_OP_IMPL_BUILDER(Key("Sigmoid").Device("GPU"), ActivationOpCudnn<float, CUDNN_ACTIVATION_SIGMOID>);
REGISTER_OP_IMPL_BUILDER(Key("Tanh").Device("GPU"),    ActivationOpCudnn<float, CUDNN_ACTIVATION_TANH>);
REGISTER_OP_IMPL_BUILDER(Key(GetGradientName("Relu")).Device("GPU"),    ActivationOpCudnnGrad<float, CUDNN_ACTIVATION_RELU>);
REGISTER_OP_IMPL_BUILDER(Key(GetGradientName("Sigmoid")).Device("GPU"), ActivationOpCudnnGrad<float, CUDNN_ACTIVATION_SIGMOID>);
REGISTER_OP_IMPL_BUILDER(Key(GetGradientName("Tanh")).Device("GPU"),    ActivationOpCudnnGrad<float, CUDNN_ACTIVATION_TANH>);

} //namespace backend
