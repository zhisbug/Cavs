#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/allocator.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/cudnn_types.h"

namespace backend {
  
using ::midend::Allocator;
using ::midend::GetAllocator;
using ::midend::DeviceTypeToString;
using ::midend::Tensor;

template <typename T>
class PoolingOpCudnn : public OpImpl {
 public:
  explicit PoolingOpCudnn(const OpDef& def);
  void Compute(OpContext* context) override;

 private:
  cudnnTensorDescriptor_t x_desc_, y_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  /*cudnnPoolingMode_t mode_;*/
  int k_;
  int stride_;
};

template <typename T>
PoolingOpCudnn<T>::PoolingOpCudnn(const OpDef& def)
  : OpImpl(def) {
  k_ = GetSingleArg<int>("k");
  stride_ = GetSingleArg<int>("stride");
  checkCUDNNError(cudnnCreateTensorDescriptor(&x_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&y_desc_));
  checkCUDNNError(cudnnCreatePoolingDescriptor(&pooling_desc_));
  checkCUDNNError(cudnnSetPooling2dDescriptor(pooling_desc_,
      CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
      k_, k_, 0, 0, stride_, stride_));
}

template <typename T>
void PoolingOpCudnn<T>::Compute(OpContext* context) {
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
  CHECK(YN == XN);
  CHECK(YC == XC);
  checkCUDNNError(cudnnSetTensor4dDescriptor(
      x_desc_,
      CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      XN, XC, XH, XW));
  checkCUDNNError(cudnnSetTensor4dDescriptor(
      y_desc_,
      CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      YN, YC, YH, YW));
  float alpha = 1.f, beta = 0.f;
  checkCUDNNError(cudnnPoolingForward(
      CudaCommon::cudnnHandle(), pooling_desc_,
      &alpha, x_desc_, x.data<T>(),
      &beta, y_desc_, y->mutable_data<T>()));
}

template <typename T>
class PoolingOpCudnnGrad : public OpImpl {
 public:
  explicit PoolingOpCudnnGrad(const OpDef& def);
  void Compute(OpContext* context) override;

 private:
  cudnnTensorDescriptor_t x_desc_, y_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  /*cudnnPoolingMode_t mode_;*/
  int k_;
  int stride_;
};

template <typename T>
PoolingOpCudnnGrad<T>::PoolingOpCudnnGrad(const OpDef& def)
  : OpImpl(def) {
  k_ = GetSingleArg<int>("k");
  stride_ = GetSingleArg<int>("stride");
  checkCUDNNError(cudnnCreateTensorDescriptor(&x_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&y_desc_));
  checkCUDNNError(cudnnCreatePoolingDescriptor(&pooling_desc_));
  checkCUDNNError(cudnnSetPooling2dDescriptor(pooling_desc_,
      CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
      k_, k_, 0, 0, stride_, stride_));
}

template <typename T>
void PoolingOpCudnnGrad<T>::Compute(OpContext* context) {
  const Tensor& y = context->Input(0);
  const Tensor& dy = context->Input(1);
  const Tensor& x = context->Input(2);
  Tensor* dx = context->Output(0);
  int YN = y.dims(0);
  int YC = y.dims(1);
  int YH = y.dims(2);
  int YW = y.dims(3);
  int XN = x.dims(0);
  int XC = x.dims(1);
  int XH = x.dims(2);
  int XW = x.dims(3);
  CHECK(YN == XN);
  CHECK(YC == XC);
  checkCUDNNError(cudnnSetTensor4dDescriptor(
      x_desc_,
      CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      XN, XC, XH, XW));
  checkCUDNNError(cudnnSetTensor4dDescriptor(
      y_desc_,
      CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      YN, YC, YH, YW));
  float alpha = 1.f, beta = 0.f;
  checkCUDNNError(cudnnPoolingBackward(
      CudaCommon::cudnnHandle(), pooling_desc_,
      &alpha, y_desc_, y.data<T>(),
      y_desc_, dy.data<T>(),
      x_desc_, x.data<T>(),
      &beta,
      x_desc_, dx->mutable_data<T>()));
}

} //namespace backend

