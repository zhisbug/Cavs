#ifndef CAVS_BACKEND_OP_IMPL_VARIABLE_H_
#define CAVS_BACKEND_OP_IMPL_VARIABLE_H_

#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/op_context.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

#include <vector>

using std::vector;

namespace backend {

using ::midend::OpContext;
using ::midend::Tensor;

template <typename FILLFUNCTOR, typename T>//fillop, dtype
class VariableOpImpl : public OpImpl {
 public:
  explicit VariableOpImpl(const OpDef& def)
    : OpImpl(def), initialized_(false) {
  }

  void Compute(OpContext* context) override {
    if (!initialized_) {
      Tensor* out = context->Output(0);
      FILLFUNCTOR(op_def_).Compute(out->mutable_data<T>(), out->count());
      //out->DebugNumerical<T>();
      initialized_ = true;
    }
    //if (!initialized_) {
      //Tensor* out = context->Output(0);
      //if (out->dims(0) == 5000) {
        //vector<float> buf(5000*100);
        //FILE *fp = fopen("/users/shizhenx/projects/tm_final/code/doc_tpc.init", "rb");
        //CHECK(fp);
        //fread(buf.data(), sizeof(float), 5000*100, fp);
        //checkCudaError(cudaMemcpy(out->mutable_data<T>(), buf.data(), 5000*100*sizeof(float),
                                  //cudaMemcpyHostToDevice));
        //fclose(fp);
        //LOG(INFO) << "doc_tpc.init...";
        ////out->DebugNumerical<float>();
        //initialized_ = true;
        //return;
      //}else if (out->dims(0) == 100) {
        //vector<float> buf(100*1000);
        //FILE *fp = fopen("/users/shizhenx/projects/tm_final/code/tpc_word.init", "rb");
        //CHECK(fp);
        //fread(buf.data(), sizeof(float), 100*1000, fp);
    
        //checkCudaError(cudaMemcpy(out->mutable_data<T>(), buf.data(), 100*1000*sizeof(float),
                                  //cudaMemcpyHostToDevice));
        //fclose(fp);
        //LOG(INFO) << "VARIABLE tpc_word.init...";
        //out->DebugNumerical<float>();
        //initialized_ = true;
        //return;
      //}
    //}
  }
 private:
  bool initialized_;
};

template <typename FILLFUNCTOR, typename T>//fillop, dtype
class DDVOpImpl : public OpImpl {
 public:
  explicit DDVOpImpl(const OpDef& def)
    : OpImpl(def), buf_(NULL), curr_idx_(-1) {
    batch_ = GetSingleArg<int>(def, "Batch");
    const std::vector<int>& shape = GetListArg<int>(def, "Shape");
    CHECK(!shape.empty());
    CHECK(shape.size() > 1);
    num_ = shape[0];
    item_size_ = 1;
    for (int i = 1; i < shape.size(); i++)
      item_size_ *= shape[i];
    CHECK(item_size_ > 0);
  }
  ~DDVOpImpl() { if (buf_) free(buf_); }

  void Compute(OpContext* context) override {
    if (!buf_) {
      buf_ = (T*)malloc(num_*item_size_*sizeof(T));
      FILLFUNCTOR(op_def_).Compute(buf_, num_*item_size_);
      //Tensor* out = context->Output(0);
      //if (num_ == 5000) {
        //vector<float> buf(5000*100);
        //FILE *fp = fopen("/users/shizhenx/projects/tm_final/code/doc_tpc.init", "rb");
        //CHECK(fp);
        //fread(buf_, sizeof(float), 5000*100, fp);
        //fclose(fp);
        //LOG(INFO) << "DDV doc_tpc.init...";
      //}
    }
    int next_idx = (context->GetRound() % (num_/batch_));
    if (next_idx != curr_idx_) {
      //LOG(INFO) << "Next idx: " << next_idx << "\tCurr idx: " << curr_idx_;
      //LOG(INFO) << "batch: " << batch_ << "\titem_size: " << item_size_;
      Tensor* out = context->Output(0);
      if (curr_idx_ >= 0) {
        checkCudaError(cudaMemcpy(buf_+curr_idx_*batch_*item_size_,
              out->mutable_data<T>(),
              out->count()*sizeof(T), 
              cudaMemcpyDeviceToHost));
      }
      CHECK(next_idx >= 0 && next_idx < num_/batch_)
        << next_idx << "\t" << num_ << "\t" << batch_;
      CHECK(out->count() == batch_*item_size_);
      checkCudaError(cudaMemcpy(out->mutable_data<T>(), 
            buf_+next_idx*batch_*item_size_,
            out->count()*sizeof(T), 
            cudaMemcpyHostToDevice));
      curr_idx_ = next_idx;
      //out->DebugNumerical<T>();
    }
  }
 private:
  int curr_idx_;
  T* buf_;
  int batch_;
  int num_;
  int item_size_;
};

} //namespace backend

#endif
