#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/allocator.h"
/*#include "cavs/midend/devices.h"*/
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/cudnn_types.h"

#include <string>
#include <vector>
#include <iomanip>

using std::string;
using std::vector;

namespace backend {

using ::midend::Allocator;
using ::midend::GetAllocator;
/*using ::midend::DeviceTypeToString;*/
using ::midend::Tensor;

template <typename T>
class RNNOpCudnnBase : public OpImpl {
 public:
  explicit RNNOpCudnnBase(const OpDef& def);
  ~RNNOpCudnnBase(); 

  virtual void InitCUDNN(const int seq_length,
      const int batch,
      const int input_size,
      const int rnn_params_count);

 protected:
  vector<cudnnTensorDescriptor_t> x_desc_ , y_desc_ ;
  cudnnTensorDescriptor_t  hx_desc_, hy_desc_;
  cudnnTensorDescriptor_t  cx_desc_, cy_desc_;
  cudnnFilterDescriptor_t  w_desc_      ;
  cudnnDropoutDescriptor_t dropout_desc_;
  cudnnRNNDescriptor_t     rnn_desc_    ;

  Allocator* alloc_;
  int hidden_size_;
  int num_layers_;
  const int num_directions_;
  string rnn_mode_ ;
  size_t rnn_workspace_sizeInBytes_;
  void* rnn_workspace_;
  size_t rnn_trainingreserve_sizeInBytes_;
  void* rnn_trainningreserve_;
  size_t rnn_hx_sizeInBytes_;
  void* rnn_hx_;
  size_t rnn_cx_sizeInBytes_;
  void* rnn_cx_;
  size_t rnn_hy_sizeInBytes_;
  void* rnn_hy_;
  size_t rnn_cy_sizeInBytes_;
  void* rnn_cy_;
  bool initialized_;

 private:
  void* dropout_workspace_;
  size_t dropout_stateSizeInBytes_;
};

template <typename T>
RNNOpCudnnBase<T>::RNNOpCudnnBase(const OpDef& def) :
    OpImpl(def),
    dropout_workspace_(NULL),
    dropout_stateSizeInBytes_(0),
    num_directions_(1),
    rnn_workspace_sizeInBytes_(0),
    rnn_trainingreserve_sizeInBytes_(0),
    rnn_workspace_(NULL), rnn_trainningreserve_(NULL),
    rnn_hx_sizeInBytes_(0), rnn_cx_sizeInBytes_(0),
    rnn_hx_(NULL), rnn_cx_(NULL),
    rnn_hy_sizeInBytes_(0), rnn_cy_sizeInBytes_(0),
    rnn_hy_(NULL), rnn_cy_(NULL),
    initialized_(false) {

  checkCUDNNError(cudnnCreateTensorDescriptor(&hx_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&hy_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&cx_desc_));
  checkCUDNNError(cudnnCreateTensorDescriptor(&cy_desc_));
  checkCUDNNError(cudnnCreateFilterDescriptor(&w_desc_));
  checkCUDNNError(cudnnCreateDropoutDescriptor(&dropout_desc_));
  checkCUDNNError(cudnnCreateRNNDescriptor(&rnn_desc_));
  alloc_ = GetAllocator(DeviceTypeToString(GPU));
  checkCUDNNError(cudnnDropoutGetStatesSize(
          CudaCommon::cudnnHandle(), &dropout_stateSizeInBytes_));
  dropout_workspace_ = alloc_->Allocate<char>(dropout_stateSizeInBytes_);
  unsigned long long SEED = 1337;
  checkCUDNNError(cudnnSetDropoutDescriptor(
        dropout_desc_,
        CudaCommon::cudnnHandle(),
        GetSingleArg<float>(def, "dropout", 0.f),
        dropout_workspace_,
        dropout_stateSizeInBytes_,
        SEED));

  hidden_size_ = GetSingleArg<int>(def, "hidden_size");
  num_layers_ = GetSingleArg<int>(def, "num_layers");
  rnn_mode_ = GetSingleArg<string>(def, "rnn_mode", "lstm");
  CHECK(rnn_mode_ == "lstm") << "Currently, we only support LSTM";
  cudnnRNNMode_t mode = CUDNN_LSTM;
  CHECK(hidden_size_ > 0);
  CHECK(num_layers_ > 0);
  checkCUDNNError(cudnnSetRNNDescriptor(
        rnn_desc_,
        hidden_size_,
        num_layers_,
        dropout_desc_,
        CUDNN_LINEAR_INPUT, //hard-coded now
        CUDNN_UNIDIRECTIONAL, //hard-coded now
        CUDNN_LSTM, //hard-coded now
        DataTypeToCudnnType<T>::value));
}

template <typename T>
RNNOpCudnnBase<T>::~RNNOpCudnnBase() {
  /*checkCUDNNError(cudnnDestroyTensorDescriptor(x_desc_));*/
  /*checkCUDNNError(cudnnDestroyTensorDescriptor(y_desc_));*/
  checkCUDNNError(cudnnDestroyTensorDescriptor(hx_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(hy_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(cx_desc_));
  checkCUDNNError(cudnnDestroyTensorDescriptor(cy_desc_));
  checkCUDNNError(cudnnDestroyFilterDescriptor(w_desc_));
  if (dropout_workspace_)
    alloc_->Deallocate<char>((char*)dropout_workspace_); 
  checkCUDNNError(cudnnDestroyDropoutDescriptor(dropout_desc_));
  checkCUDNNError(cudnnDestroyRNNDescriptor(rnn_desc_));
  if (!x_desc_.empty()) {
    for (auto& des : x_desc_)
      checkCUDNNError(cudnnDestroyTensorDescriptor(des));
  }
  if (!y_desc_.empty()) {
    for (auto& des : y_desc_)
      checkCUDNNError(cudnnDestroyTensorDescriptor(des));
  }
  if (rnn_workspace_)
    alloc_->Deallocate<char>((char*)rnn_workspace_); 
  if (rnn_trainningreserve_)
    alloc_->Deallocate<char>((char*)rnn_trainningreserve_); 
  if (rnn_hx_)
    alloc_->Deallocate<char>((char*)rnn_hx_); 
  if (rnn_cx_)
    alloc_->Deallocate<char>((char*)rnn_cx_); 
  if (rnn_hy_)
    alloc_->Deallocate<char>((char*)rnn_hy_); 
  if (rnn_cy_)
    alloc_->Deallocate<char>((char*)rnn_cy_); 
}

template <typename T>
void RNNOpCudnnBase<T>::InitCUDNN(
    const int seq_length,
    const int batch,
    const int input_size,
    const int rnn_params_count) {
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
    //I think there is a problem in the first dimension of dim
    //Should it be num_layers? and the third dimension is hidden*directions?
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

    size_t internal_size = num_layers_*num_directions_*batch*hidden_size_*sizeof(T);
    if (rnn_hx_sizeInBytes_ != internal_size) {
      vector<T> test;
      /*test.resize(internal_size/sizeof(T), 1);*/
      if (rnn_hx_) 
        alloc_->Deallocate<char>((char*)rnn_hx_); 
      rnn_hx_ = alloc_->Allocate<char>(internal_size);
      /*checkCudaError(cudaMemcpy(rnn_hx_, test.data(), internal_size, cudaMemcpyHostToDevice));*/
      rnn_hx_sizeInBytes_ = internal_size; 
    }
    if (rnn_cx_sizeInBytes_ != internal_size) {
      vector<T> test;
      /*test.resize(internal_size/sizeof(T), 1);*/
      if (rnn_cx_) 
        alloc_->Deallocate<char>((char*)rnn_cx_); 
      rnn_cx_ = alloc_->Allocate<char>(internal_size);
      /*checkCudaError(cudaMemcpy(rnn_cx_, test.data(), internal_size, cudaMemcpyHostToDevice));*/
      rnn_cx_sizeInBytes_ = internal_size; 
    }
    if (rnn_hy_sizeInBytes_ != internal_size) {
      if (rnn_hy_) 
        alloc_->Deallocate<char>((char*)rnn_hy_); 
      rnn_hy_ = alloc_->Allocate<char>(internal_size);
      rnn_hy_sizeInBytes_ = internal_size; 
    }
    if (rnn_cy_sizeInBytes_ != internal_size) {
      if (rnn_cy_) 
        alloc_->Deallocate<char>((char*)rnn_cy_); 
      rnn_cy_ = alloc_->Allocate<char>(internal_size);
      rnn_cy_sizeInBytes_ = internal_size; 
    }
  }

  {
    size_t rnn_params_sizeInBytes;
    checkCUDNNError(cudnnGetRNNParamsSize(
          CudaCommon::cudnnHandle(),
          rnn_desc_,
          x_desc_[0],
          &rnn_params_sizeInBytes,
          DataTypeToCudnnType<T>::value));
    CHECK(rnn_params_count == rnn_params_sizeInBytes/sizeof(T))
        << "Input variable count : " << rnn_params_count
        << "\t CUDNN needs : " << rnn_params_sizeInBytes/sizeof(T);
    const std::array<int, 3> dim = {(int)(rnn_params_sizeInBytes/sizeof(T)), 1, 1};
    checkCUDNNError(cudnnSetFilterNdDescriptor(
          w_desc_,
          DataTypeToCudnnType<T>::value,
          CUDNN_TENSOR_NCHW,
          3,
          dim.data()));
  }

  {
    size_t workspace_size;
    checkCUDNNError(cudnnGetRNNWorkspaceSize(
          CudaCommon::cudnnHandle(),
          rnn_desc_,
          seq_length,
          x_desc_.data(),
          &workspace_size)); 
    if (workspace_size != rnn_workspace_sizeInBytes_) {
      rnn_workspace_sizeInBytes_ = workspace_size; 
      if (rnn_workspace_)
        alloc_->Deallocate<char>((char*)rnn_workspace_); 
      rnn_workspace_ = alloc_->Allocate<char>(rnn_workspace_sizeInBytes_);
    }
  }
}

template <typename T>
class RNNOpCudnn: public RNNOpCudnnBase<T> {
 public:
  explicit RNNOpCudnn(const OpDef& def);
  void Compute(OpContext* context) override;
};

template <typename T>
RNNOpCudnn<T>::RNNOpCudnn(const OpDef& def)
  : RNNOpCudnnBase<T>(def) {}

template <typename T>
void RNNOpCudnn<T>::Compute(OpContext* context) {
  const Tensor& X = context->Input(0);
  const Tensor& W = context->Input(1);
  /*const Tensor& HX = context->Input(2);*/
  /*const Tensor& CX = context->Input(3);*/
  Tensor* Y = context->Output(0);
  /*Tensor* HY = context->Output(1);*/
  /*Tensor* CY = context->Output(2);*/

  const int seq_length = X.dims(0);
  const int batch      = X.dims(1);
  const int input_size = X.dims(2);
  const int rnn_params_count = W.count();

  if (!this->initialized_) {
    this->InitCUDNN(seq_length, batch, input_size, rnn_params_count);
    cudnnFilterDescriptor_t  desc;
    checkCUDNNError(cudnnCreateFilterDescriptor(&desc));
    for (int l = 0; l < this->num_layers_; l++) {
      for (int g = 0; g < 8; g++) {
        void *buf = NULL;
        checkCUDNNError(cudnnGetRNNLinLayerBiasParams(
              CudaCommon::cudnnHandle(),
              this->rnn_desc_,
              l,
              this->x_desc_[0],
              this->w_desc_,
              context->Input(1).mutable_data<T>(),
              g,
              desc,
              &buf));
        cudnnDataType_t dt;
        cudnnTensorFormat_t tf;
        vector<int> dims(3);
        int nbDims;
        checkCUDNNError(cudnnGetFilterNdDescriptor(desc,
              3, &dt, &tf, &nbDims, dims.data()));
        LOG(INFO) << "Init bias...";
        int count = 1;
        for (auto i : dims) {
          count *= i;
          LOG(INFO) << i;
        }
        CHECK(buf);
        if (false && (g == 1 || g == 5))
          checkCudaError(cudaMemset(buf, 0, count*sizeof(T)));
      }
    }
    checkCUDNNError(cudnnDestroyFilterDescriptor(desc));
    this->initialized_ = true;
  }

  {
    size_t workspace_size;
    checkCUDNNError(cudnnGetRNNTrainingReserveSize(
          CudaCommon::cudnnHandle(),
          this->rnn_desc_,
          seq_length,
          this->x_desc_.data(),
          &workspace_size)); 
    if (workspace_size != this->rnn_trainingreserve_sizeInBytes_) {
      this->rnn_trainingreserve_sizeInBytes_ = workspace_size; 
      if (this->rnn_trainningreserve_)
        this->alloc_->template Deallocate<char>((char*)(this->rnn_trainningreserve_)); 
      this->rnn_trainningreserve_ = (this->alloc_)->template Allocate<char>(this->rnn_trainingreserve_sizeInBytes_);
      context->repo_[Y->name()] = this->rnn_trainningreserve_;
    }
  }

  /*{*/
    /*vector<float> testW;*/
    /*testW.resize(W.count(), 0.01f);*/
    /*checkCudaError(cudaMemcpy(context->Input(1).mutable_data<T>(), testW.data(), W.count()*sizeof(float), cudaMemcpyHostToDevice));*/
    /*vector<float> testX;*/
    /*testX.resize(X.count(), 1.f);*/
    /*checkCudaError(cudaMemcpy(context->Input(0).mutable_data<T>(), testX.data(), X.count()*sizeof(float), cudaMemcpyHostToDevice));*/
  /*}*/

  checkCUDNNError(cudnnRNNForwardTraining(
        CudaCommon::cudnnHandle(),
        this->rnn_desc_,
        seq_length,
        this->x_desc_.data(),
        X.data<T>(),
        this->hx_desc_,
        nullptr,// this->rnn_hx_, //HX.data<T>(),
        this->cx_desc_,
        nullptr,// this->rnn_cx_, //CX.data<T>(),
        this->w_desc_,
        W.data<T>(),
        this->y_desc_.data(),
        Y->mutable_data<T>(),
        this->hy_desc_,
        nullptr, //this->rnn_hy_, //HY->mutable_data<T>(),
        this->cy_desc_,
        nullptr, //this->rnn_cy_, //CY->mutable_data<T>(),
        this->rnn_workspace_,
        this->rnn_workspace_sizeInBytes_,
        this->rnn_trainningreserve_,
        this->rnn_trainingreserve_sizeInBytes_));

  X.DebugNumerical<T>();
  W.DebugNumerical<T>();
  Y->DebugNumerical<T>();
}

template <typename T>
class RNNOpCudnnGrad: public RNNOpCudnnBase<T> {
 public:
  explicit RNNOpCudnnGrad(const OpDef& def);
  void Compute(OpContext* context) override;
};

template <typename T>
RNNOpCudnnGrad<T>::RNNOpCudnnGrad(const OpDef& def)
  : RNNOpCudnnBase<T>(def) {}

template <typename T>
void RNNOpCudnnGrad<T>::Compute(OpContext* context) {
  const Tensor& Y  = context->Input(0);
  const Tensor& dY = context->Input(1);
  const Tensor& X  = context->Input(2);
  const Tensor& W  = context->Input(3);
  /*const Tensor& HX = context->Input(4);*/
  /*const Tensor& CX = context->Input(5);*/

  Tensor* dX  = context->Output(0);
  Tensor* dW  = context->Output(1);
  /*Tensor* dHX = context->Output(2);*/
  /*Tensor* dCX = context->Output(3);*/

  const int seq_length = X.dims(0);
  const int batch      = X.dims(1);
  const int input_size = X.dims(2);
  const int rnn_params_count = W.count();

  if (!this->initialized_) {
    this->InitCUDNN(seq_length, batch, input_size, rnn_params_count);
    this->initialized_ = true;
  }

  {
    size_t workspace_size;
    checkCUDNNError(cudnnGetRNNTrainingReserveSize(
          CudaCommon::cudnnHandle(),
          this->rnn_desc_,
          seq_length,
          this->x_desc_.data(),
          &workspace_size)); 
    if (workspace_size != this->rnn_trainingreserve_sizeInBytes_) {
      this->rnn_trainingreserve_sizeInBytes_ = workspace_size; 
      CHECK(context->repo_.find(Y.name()) != context->repo_.end());
      this->rnn_trainningreserve_ = context->repo_[Y.name()];
      CHECK(this->rnn_trainningreserve_);
    }
  }

  /*{*/
    /*vector<float> testdY;*/
    /*testdY.resize(dY.count(), .3f);*/
    /*checkCudaError(cudaMemcpy(context->Input(1).mutable_data<T>(), testdY.data(), dY.count()*sizeof(float), cudaMemcpyHostToDevice));*/
  /*}*/

  checkCUDNNError(cudnnRNNBackwardData(
        CudaCommon::cudnnHandle(),
        this->rnn_desc_,
        seq_length,
        this->y_desc_.data(),
        Y.data<T>(),
        this->y_desc_.data(),
        dY.data<T>(),
        this->hy_desc_,
        nullptr,//dhy can be nullptr, that means 0 according to cudnn manual
        this->cy_desc_,
        nullptr,//dcy can be nullptr, that means 0 according to cudnn manual
        this->w_desc_,
        W.data<T>(),
        this->hx_desc_,
        nullptr, //this->rnn_hx_, //HX.data<T>(),
        this->cx_desc_,
        nullptr, //this->rnn_cx_, //CX.data<T>(),
        this->x_desc_.data(),
        dX->mutable_data<T>(),
        this->hx_desc_,
        nullptr,//this->rnn_hy_, //hacked here //dHX->mutable_data<T>(),
        this->cx_desc_,
        nullptr,//this->rnn_cy_, //hacked here //dCX->mutable_data<T>(),
        this->rnn_workspace_,
        this->rnn_workspace_sizeInBytes_,
        this->rnn_trainningreserve_,
        this->rnn_trainingreserve_sizeInBytes_));

  checkCudaError(cudaMemset(dW->mutable_data<T>(), 0, dW->count()*sizeof(T)));

  checkCUDNNError(cudnnRNNBackwardWeights(
        CudaCommon::cudnnHandle(),
        this->rnn_desc_,
        seq_length,
        this->x_desc_.data(),
        X.data<T>(),
        this->hx_desc_,
        nullptr, //this->rnn_hx_, //HX.data<T>(),
        this->y_desc_.data(),
        Y.data<T>(),
        this->rnn_workspace_,
        this->rnn_workspace_sizeInBytes_,
        this->w_desc_,
        dW->mutable_data<T>(),
        this->rnn_trainningreserve_,
        this->rnn_trainingreserve_sizeInBytes_));

  if (false && VLOG_IS_ON(V_EXHAUSTIVE_DEBUG)) {
    cudnnFilterDescriptor_t  desc;
    void* buf;
    VLOG(V_DEBUG)  << "address of dW:" << std::dec << dW->mutable_data<T>();
    for (int i = 0; i < 8; i++) {
      checkCUDNNError(cudnnCreateFilterDescriptor(&desc));
      checkCUDNNError(cudnnGetRNNLinLayerMatrixParams(
            CudaCommon::cudnnHandle(),
            this->rnn_desc_,
            1,
            this->x_desc_[0],
            this->w_desc_,
            dW->mutable_data<T>(),
            i,
            desc,
            &buf));
      VLOG(V_DEBUG)  << "address of matrix layer[" << i << "]: " << std::dec << buf;
    }

    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 8; i++) {
        checkCUDNNError(cudnnGetRNNLinLayerBiasParams(
              CudaCommon::cudnnHandle(),
              this->rnn_desc_,
              j,
              this->x_desc_[0],
              this->w_desc_,
              dW->mutable_data<T>(),
              i,
              desc,
              &buf));
        VLOG(V_DEBUG)  << "address of bias layer[" << j << "][" << i << "]: " << std::dec << buf;
      }
  }
  Y.DebugNumerical<T>();
  dY.DebugNumerical<T>();
  X.DebugNumerical<T>();
  W.DebugNumerical<T>();
  dX->DebugNumerical<T>();
  dW->DebugNumerical<T>();
}

REGISTER_OP_IMPL_BUILDER(Key("LSTM").Device("GPU"), RNNOpCudnn<float>);
REGISTER_OP_IMPL_BUILDER(Key(GetGradientName("LSTM")).Device("GPU"), RNNOpCudnnGrad<float>);

} //namespace backend
