#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/cudnn_types.h"

namespace backend {

using ::midend::Tensor;

class SoftmaxEntropyLogitsOpCudnnBase : public OpImpl {
 public:
  explicit SoftmaxEntropyLogitsOpCudnnBase(const OpDef& def);
  ~SoftmaxEntropyLogitsOpCudnnBase();

 protected:
  cudnnTensorDescriptor_t x_desc_, y_desc_;
  cudnnTensorDescriptor_t label_desc_;
};

SoftmaxEntropyLogitsOpCudnnBase::SoftmaxEntropyLogitsOpCudnnBase(const OpDef& def)
    : OpImpl(def) {
  checkCUDNNError(cudnnCreateTensorDescriptor(&x_desc_));    
  checkCUDNNError(cudnnCreateTensorDescriptor(&y_desc_));    
  checkCUDNNError(cudnnCreateTensorDescriptor(&label_desc_));    
}

SoftmaxEntropyLogitsOpCudnnBase::~SoftmaxEntropyLogitsOpCudnnBase() {
  checkCUDNNError(cudnnDestroyTensorDescriptor(x_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(y_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(label_desc_));
}

template <typename T>
class SoftmaxEntropyLogitsOpCudnn : public SoftmaxEntropyLogitsOpCudnnBase {
 public:
  explicit SoftmaxEntropyLogitsOpCudnn(const OpDef& def) 
      : SoftmaxEntropyLogitsOpCudnnBase(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
void SoftmaxEntropyLogitsOpCudnn<T>::Compute(OpContext* context) {
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
  checkCUDNNError(cudnnSoftmaxForward(CudaCommon::cudnnHandle(),
                                      CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &alpha, x_desc_, x.data<T>(),
                                      &beta, y_desc_, y->mutable_data<T>()));
}

template <typename T>
class SoftmaxEntropyLogitsOpCudnnGrad : public SoftmaxEntropyLogitsOpCudnnBase {
 public:
  explicit SoftmaxEntropyLogitsOpCudnnGrad(const OpDef& def) 
      : SoftmaxEntropyLogitsOpCudnnBase(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
__global__ void SoftmaxEntropyLogitsBackwardKernel(T* dx, 
    const T* y, const T* label,
    int elements, int prediction_range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements)  return;
  const int label_value = static_cast<int>(label[idx/prediction_range]);
  //Through the formula of cross-entropy,
  //the derivation of dx can be denoted as follows:
  //(I deduce it for a whole noon!)
  if (label_value == idx%prediction_range)
    dx[idx] = y[idx] - 1;
  else
    dx[idx] = y[idx];
}

template <typename T>
void SoftmaxEntropyLogitsOpCudnnGrad<T>::Compute(OpContext* context) {
  const Tensor& y = context->Input(0);
  const Tensor& label = context->Input(1);
  Tensor* dx = context->Output(0);

  int NY = y.dims(0);
  int CY = y.dims(1);
  int NLabel = label.dims(0);
  int CLabel = label.dims(1);
  int NX = dx->dims(0);
  int CX = dx->dims(1);
  CHECK(NY == NX == NLabel);
  CHECK(CY == CX);
  CHECK(CLabel == 1);
  CHECK(y.dims(2) == y.dims(3) == label.dims(2) == label.dims(3)
        == dx->dims(2) == dx->dims(3) == 1);

  /*float alpha = 1.f, beta = 0.f;*/
  /*checkCUDNNError(cudnnSetTensor4dDescriptor(x_desc_,*/
                  /*CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,*/
                  /*x.dims(0), x.dims(1), x.dims(2), x.dims(3)));*/
  /*checkCUDNNError(cudnnSetTensor4dDescriptor(label_desc_,*/
                  /*CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,*/
                  /*label.dims(0), label.dims(1), label.dims(2), label.dims(3)));*/
  /*checkCUDNNError(cudnnSetTensor4dDescriptor(y_desc_,*/
                  /*CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,*/
                  /*y.dims(0), y.dims(1), y.dims(2), y.dims(3)));*/
  int n = y.count();
  SoftmaxEntropyLogitsBackwardKernel<T><<<THREADS_PER_BLOCK, BLOCKS_PER_GRID(n)>>>(
        dx->mutable_data<T>(), y.data<T>(), label.data<T>(), n, CY);
  checkCudaError(cudaDeviceSynchronize());
}

} //namespace backend

