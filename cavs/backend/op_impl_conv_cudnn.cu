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

class ConvOpCudnnBase : public OpImpl {
 public:
  explicit ConvOpCudnnBase(const OpDef& def);
  ~ConvOpCudnnBase(); 

 protected:
  cudnnTensorDescriptor_t x_desc_, y_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_f_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_d_algo_;
  Allocator* alloc_;
};

ConvOpCudnnBase::ConvOpCudnnBase(const OpDef& def)
    : OpImpl(def) {
  checkCUDNNError(cudnnCreateTensorDescriptor(&x_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&y_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&bias_desc_));
  checkCUDNNError(cudnnCreateFilterDescriptor(&filter_desc_));
  checkCUDNNError(cudnnCreateConvolutionDescriptor(&conv_desc_));
  alloc_ = GetAllocator(DeviceTypeToString(GPU));
}

ConvOpCudnnBase::~ConvOpCudnnBase() {
  checkCUDNNError(cudnnDestroyTensorDescriptor(x_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(y_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(bias_desc_));
  checkCUDNNError(cudnnDestroyFilterDescriptor(filter_desc_));
  checkCUDNNError(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

template <typename T>
class ConvOpCudnn: public ConvOpCudnnBase {
 public:
  explicit ConvOpCudnn(const OpDef& def) 
      : ConvOpCudnnBase(def),
        workspace(NULL), workspaceSizeInBytes(0) {}
  ~ConvOpCudnn();
  void Compute(OpContext* context) override;
  /*static void inference_shape*/

 private:
  size_t workspaceSizeInBytes;
  void* workspace;
};

template <typename T>
ConvOpCudnn<T>::~ConvOpCudnn() { 
  if (workspace)
    alloc_->Deallocate<char>((char*)workspace); 
}

template <typename T>
void ConvOpCudnn<T>::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  const Tensor& filter = context->Input(1);
  const Tensor& bias   = context->Input(2);
  Tensor* y = context->Output(0);

  int XN = x.dims(0);
  int XC = x.dims(1);
  int XH = x.dims(2);
  int XW = x.dims(3);
  int FYC = filter.dims(0);
  int FXC = filter.dims(1);
  int FH = filter.dims(2);
  int FW = filter.dims(3);
  int YN = y->dims(0);
  int YC = y->dims(1);
  int YH = y->dims(2);
  int YW = y->dims(3);
  CHECK(FXC == XC);
  CHECK(FYC == YC);
  CHECK(YN == XN);

  checkCUDNNError(cudnnSetTensor4dDescriptor(bias_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  1, FYC, 1, 1));
  checkCUDNNError(cudnnSetTensor4dDescriptor(x_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  XN, XC, XH, XW));
  checkCUDNNError(cudnnSetTensor4dDescriptor(y_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  YN, YC, YH, YW));
  checkCUDNNError(cudnnSetFilter4dDescriptor(filter_desc_,
                  DataTypeToCudnnType<T>::value, 
                  FYC, FXC, FH, FW));
  checkCUDNNError(cudnnSetConvolution2dDescriptor(conv_desc_,
                  0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
  /*checkCUDNNError(cudnnGetConvolutionNdForwardOutputDim(*/
                  /*conv_desc_, x_desc_, filter_desc_, */
                  /*4, YDim));*/

  checkCUDNNError(cudnnGetConvolutionForwardAlgorithm(CudaCommon::cudnnHandle(),
                  x_desc_, filter_desc_, conv_desc_, y_desc_,
                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo_));
  checkCUDNNError(cudnnGetConvolutionForwardWorkspaceSize(CudaCommon::cudnnHandle(),
                  x_desc_, filter_desc_, conv_desc_, y_desc_,
                  fwd_algo_, &workspaceSizeInBytes));
  /*checkCudaError(cudaMalloc((void**)&workspace, workspaceSizeInBytes));*/
  if (workspace)
    alloc_->Deallocate<char>((char*)workspace);
  workspace = alloc_->Allocate<char>(workspaceSizeInBytes);

  float alpha = 1.f, beta = 0.f;
  checkCUDNNError(cudnnConvolutionForward(CudaCommon::cudnnHandle(), 
                  &alpha, x_desc_,
                  x.data<T>(), filter_desc_, filter.data<T>(),
                  conv_desc_, fwd_algo_, workspace, workspaceSizeInBytes, &beta,
                  y_desc_, y->mutable_data<T>()));
  checkCUDNNError(cudnnAddTensor(CudaCommon::cudnnHandle(),
                  &alpha, bias_desc_,
                  bias.data<T>(), &alpha, 
                  y_desc_, y->mutable_data<T>()));
}

template <typename T>
class ConvOpCudnnGrad: public ConvOpCudnnBase {
 public:
  explicit ConvOpCudnnGrad(const OpDef& def) 
      : ConvOpCudnnBase(def), 
      filter_workspace(NULL), data_workspace(NULL),
      filter_workspaceSizeInBytes(0), data_workspaceSizeInBytes(0) {}
  ~ConvOpCudnnGrad(); 
  void Compute(OpContext* context) override;

 private:
  size_t filter_workspaceSizeInBytes;
  size_t data_workspaceSizeInBytes;
  void* filter_workspace;
  void* data_workspace;
};

template <typename T>
ConvOpCudnnGrad<T>::~ConvOpCudnnGrad() { 
  alloc_->Deallocate<char>((char*)filter_workspace); 
  alloc_->Deallocate<char>((char*)data_workspace); 
}

template <typename T>
void ConvOpCudnnGrad<T>::Compute(OpContext* context) {
  const Tensor& dy = context->Input(0);
  const Tensor& x = context->Input(1);
  const Tensor& filter = context->Input(2);
  const Tensor& bias   = context->Input(3);
  Tensor* df = context->Output(0);
  Tensor* dx = context->Output(1);
  Tensor* db = context->Output(2);

  int XN = x.dims(0);
  int XC = x.dims(1);
  int XH = x.dims(2);
  int XW = x.dims(3);
  int FYC = filter.dims(0);
  int FXC = filter.dims(1);
  int FH = filter.dims(2);
  int FW = filter.dims(3);
  int YN = dy.dims(0);
  int YC = dy.dims(1);
  int YH = dy.dims(2);
  int YW = dy.dims(3);
  CHECK(FXC == XC);
  CHECK(FYC == YC);
  CHECK(YN == XN);

  checkCUDNNError(cudnnSetTensor4dDescriptor(bias_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  1, FYC, 1, 1));
  checkCUDNNError(cudnnSetTensor4dDescriptor(x_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  XN, XC, XH, XW));
  checkCUDNNError(cudnnSetTensor4dDescriptor(y_desc_,
                  CUDNN_TENSOR_NCHW, DataTypeToCudnnType<T>::value,
                  YN, YC, YH, YW));
  checkCUDNNError(cudnnSetFilter4dDescriptor(filter_desc_,
                  DataTypeToCudnnType<T>::value, 
                  FYC, FXC, FH, FW));
  checkCUDNNError(cudnnSetConvolution2dDescriptor(conv_desc_,
                  0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
  checkCUDNNError(cudnnGetConvolutionBackwardFilterAlgorithm(CudaCommon::cudnnHandle(),
                  x_desc_, y_desc_, conv_desc_, filter_desc_, 
                  CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwd_f_algo_));
  checkCUDNNError(cudnnGetConvolutionBackwardFilterWorkspaceSize(CudaCommon::cudnnHandle(),
                  x_desc_, y_desc_, conv_desc_, filter_desc_,
                  bwd_f_algo_, &filter_workspaceSizeInBytes));
  checkCUDNNError(cudnnGetConvolutionBackwardDataAlgorithm(CudaCommon::cudnnHandle(),
                  filter_desc_, y_desc_, conv_desc_, x_desc_,
                  CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_d_algo_));
  checkCUDNNError(cudnnGetConvolutionBackwardDataWorkspaceSize(CudaCommon::cudnnHandle(),
                  filter_desc_, y_desc_, conv_desc_, x_desc_,
                  bwd_d_algo_, &data_workspaceSizeInBytes));
  if (filter_workspace)
    alloc_->Deallocate<char>((char*)filter_workspace);
  filter_workspace = alloc_->Allocate<char>(filter_workspaceSizeInBytes);
  if (data_workspace)
    alloc_->Deallocate<char>((char*)data_workspace);
  data_workspace = alloc_->Allocate<char>(data_workspaceSizeInBytes);

  float alpha = 1.f, beta = 0.f;
  checkCUDNNError(cudnnConvolutionBackwardFilter(CudaCommon::cudnnHandle(), 
                  &alpha, x_desc_, x.data<T>(),
                  y_desc_, dy.data<T>(),
                  conv_desc_, bwd_f_algo_, filter_workspace, filter_workspaceSizeInBytes,
                  &beta, filter_desc_, df->mutable_data<T>()));
  checkCUDNNError(cudnnConvolutionBackwardData(CudaCommon::cudnnHandle(),
                  &alpha, filter_desc_, filter.data<T>(), 
                  y_desc_, dy.data<T>(),
                  conv_desc_, bwd_d_algo_, data_workspace, data_workspaceSizeInBytes,
                  &beta, x_desc_, x.mutable_data<T>()));
  checkCUDNNError(cudnnConvolutionBackwardBias(CudaCommon::cudnnHandle(), 
                  &alpha, y_desc_, dy.data<T>(),
                  &beta, bias_desc_, db->mutable_data<T>()));
}

} //namespace backend
