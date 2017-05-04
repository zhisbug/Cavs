#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/allocator.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/cudnn_types.h"

#include <string>
#include <vector>

using std::string;
using std::vector;

namespace backend {

using ::midend::Allocator;
using ::midend::GetAllocator;
using ::midend::DeviceTypeToString;
using ::midend::Tensor;

class RNNOpCudnnBase : public OpImpl {
 public:
  explicit RNNOpCudnnBase(const OpDef& def);
  ~RNNOpCudnnBase(); 

 protected:
  vector<cudnnTensorDescriptor_t> x_desc_ , y_desc_ ;
  cudnnTensorDescriptor_t  hx_desc_, hy_desc_;
  cudnnTensorDescriptor_t  cx_desc_, cy_desc_;
  cudnnFilterDescriptor_t  w_desc_      ;
  cudnnDropoutDescriptor_t dropout_desc_;
  cudnnRNNDescriptor_t     rnn_desc_    ;

  Allocator* alloc_;
 private:
  void* dropout_workspace_;
};

RNNOpCudnnBase::RNNOpCudnnBase(const OpDef& def) :
    OpImpl(def),
    dropout_workspace(NULL),
    dropout_workspaceSizeInBytes(0) {

  /*checkCUDNNError(cudnnCreateTensorDescriptor(&x_desc_));*/
  /*checkCUDNNError(cudnnCreateTensorDescriptor(&y_desc_));*/
  checkCUDNNError(cudnnCreateTensorDescriptor(&hx_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&hy_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&cx_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&cy_desc_));
  checkCUDNNError(cudnnCreateFilterDescriptor(&w_desc_));
  checkCUDNNError(cudnnCreateDropoutDescriptor(&dropout_desc_));
  checkCUDNNError(cudnnCreateRNNDescriptor(&rnn_desc_));

  alloc_ = GetAllocator(DeviceTypeToString(GPU));

  size_t dropout_stateSizeInBytes;
  checkCUDNNError(cudnnDropoutGetStatesSize(
          CudaCommon::cudnnHandle(), &dropout_stateSizeInBytes));
  dropout_workspace_ = alloc_->Allocate<char>(dropout_stateSizeInBytes);
  unsigned long long SEED = 1337;
  checkCUDNNError(cudnnSetDropoutDescriptor(
        dropout_desc_,
        CudaCommon::cudnnHandle(),
        GetSingleArg<float>(def, "dropout", 1.f),
        dropout_workspace_,
        dropout_stateSizeInBytes,
        SEED));
}

RNNOpCudnnBase::~RNNOpCudnnBase() {
  checkCUDNNError(cudnnDestroyTensorDescriptor(x_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(y_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(hx_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(hy_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(cx_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(cy_desc_));
  checkCUDNNError(cudnnDestroyFilterDescriptor(w_desc_));
  if (dropout_workspace_)
    alloc_->Deallocate<char>((char*)dropout_workspace_); 
  checkCUDNNError(cudnnDestroyDropoutDescriptor(dropout_desc_));
  checkCUDNNError(cudnnDestroyRNNDescriptor(rnn_desc_));

}

template <typename T>
class RNNOpCudnn: public RNNOpCudnnBase {
 public:
  explicit RNNOpCudnn(const OpDef& def);
  ~RNNOpCudnn();
  void Compute(OpContext* context) override;
  /*static void inference_shape*/

 private:
  int hidden_size_;
  int num_layers_;
  const int num_directions_;
  string rnn_mode_ ;
};

template <typename T>
RNNOpCudnn<T>::RNNOpCudnn() :
    RNNOpCudnnBase(def), num_directions_(1) {
  hidden_size_ = GetSingleArg<int>(def, "hidden_size");
  rnn_mode_ = GetSingleArg<string>("rnn_mode", "lstm");
  num_layers_ = OperatorBase::GetSingleArgument<int>("num_layers", 0);
  CHECK(rnn_mode_ == "lstm") << "Currently, we only support LSTM";
  cudnnRNNMode_t mode = CUDNN_LSTM;
  CHECK(hidden_size_ > 0);
  checkCUDNNError(cudnnSetRNNDescriptor(
        rnn_desc_,
        hidden_size_,
        num_layers_,
        dropout_desc_,
        CUDNN_LSTM, //hard-coded now
        CUDNN_LINEAR_INPUT, //hard-coded now
        CUDNN_UNIDIRECTIONAL, //hard-coded now
        CUDNN_LSTM, //hard-coded now
        DataTypeToCudnnType<T>::value));
}

template <typename T>
RNNOpCudnn<T>::~RNNOpCudnn() { 
  if (workspace)
    alloc_->Deallocate<char>((char*)workspace); 
  if (!x_desc_.empty()) {
    for (auto& des : x_desc_)
      checkCUDNNError(cudnnDestroyTensorDescriptor(des));
  }
  if (!y_desc_.empty()) {
    for (auto& des : y_desc_)
      checkCUDNNError(cudnnDestroyTensorDescriptor(des));
  }
}

template <typename T>
void RNNOpCudnn<T>::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  const Tensor& w = context->Input(1);

  const int seq_length = x.dims(0);
  const int batch      = x.dims(1);
  const int input_size = x.dims(2);
  CHECK(x_desc_.empty() || x_desc_.size() == seq_length)
       << "only support fixed size corpus during iterations";
  CHECK(y_desc_.empty() || y_desc_.size() == seq_length)
       << "only support fixed size corpus during iterations";
  CHECK(seq_length > 0);
  if (x_desc_.empty()) {
    x_desc_.resize(seq_length); 
    const std::array<int, 3> dim = {batch, input_size, 1};
    const std::array<int, 3> stride = {input_size, 1, 1};
    for (int i = 0; i < seq_length; i++) {
      checkCUDNNError(cudnnCreateTensorDescriptor(&x_desc_[i]));  
      checkCUDNNError(cudnnSetTensorNdDescriptor(
            x_desc_[i], DataTypeToCudnnType<T>::value, 3, dim.data(), stride.data()));
    }
  }

  if (y_desc_.empty()) {
    y_desc_.resize(seq_length); 
    const std::array<int, 3> dim = {batch, hidden_size_*num_directions_, 1};
    const std::array<int, 3> stride = {hidden_size_*num_directions_, 1, 1};
    for (int i = 0; i < seq_length; i++) {
      checkCUDNNError(cudnnCreateTensorDescriptor(&y_desc_[i]));  
      checkCUDNNError(cudnnSetTensorNdDescriptor(
            y_desc_[i], DataTypeToCudnnType<T>::value, 3, dim.data(), stride.data()));
    }
  }
  
  {
    const std::array<int, 3> dim = {num_layers_*num_directions_, batch, hidden_size_};
    const std::array<int, 3> stride = {batch*hidden_size_, hidden_size_, 1};
    checkCUDNNError(cudnnSetTensorNdDescriptor(
          hx_desc_, DataTypeToCudnnType<T>::value, 3, dim.data(), stride.data()));
    checkCUDNNError(cudnnSetTensorNdDescriptor(
          hy_desc_, DataTypeToCudnnType<T>::value, 3, dim.data(), stride.data()));
    checkCUDNNError(cudnnSetTensorNdDescriptor(
          cx_desc_, DataTypeToCudnnType<T>::value, 3, dim.data(), stride.data()));
    checkCUDNNError(cudnnSetTensorNdDescriptor(
          cy_desc_, DataTypeToCudnnType<T>::value, 3, dim.data(), stride.data()));
  }

  {
    size_t rnn_params_sizeInBytes;
    checkCUDNNError(cudnnGetRNNParamsSize(
          CudaCommon::cudnnHandle(),
          rnn_desc_,
          x_desc_[0],
          &rnn_params_sizeInBytes,
          DataTypeToCudnnType<T>::value));
    const std::array<int, 3> dim = {rnn_params_sizeInBytes/sizeof(T), 1, 1};
    checkCUDNNError(cudnnSetFilterNdDescriptor(
          w_desc_,
          DataTypeToCudnnType<T>::value,
          CUDNN_TENSOR_NCHW,
          3,
          dim.data()));
    CHECK(w.count() == rnn_params_sizeInBytes/sizeof(T));
  }
}

