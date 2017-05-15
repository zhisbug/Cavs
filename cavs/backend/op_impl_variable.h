#ifndef CAVS_BACKEND_OP_IMPL_VARIABLE_H_
#define CAVS_BACKEND_OP_IMPL_VARIABLE_H_

#include "cavs/backend/op_impl.h"
#include "cavs/backend/op_impl_mpi_functor.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/op_context.h"
#include "cavs/midend/tensor.h"

#include <vector>
#include <random>

using std::vector;

namespace backend {

using ::midend::OpContext;
using ::midend::Tensor;

template <typename FILLFUNCTOR, typename T, typename BCASTFUNCTOR=bool>//fillop, dtype
class VariableOpImpl : public OpImpl {
 public:
  explicit VariableOpImpl(const OpDef& def)
    : OpImpl(def), initialized_(false) {}
  void Compute(OpContext* context) override;

 private:
  bool initialized_;
};

template <typename FILLFUNCTOR, typename T, bool MPIEnable>//fillop, dtype
class DDVOpImpl : public OpImpl {
 public:
  explicit DDVOpImpl(const OpDef& def);
  ~DDVOpImpl();
  void Compute(OpContext* context) override;

 private:
  int curr_idx_;
  T* buf_;
  int batch_;
  int num_;
  int item_size_;
};

template <typename T>
struct HasBcast {
 private:
  template<typename U, void (*)(void*, int, int)>
  struct matcher;

  template <typename U>
  static char helper(matcher<U, &U::Compute>*);

  template <typename U>
  static int helper(...);

 public:
  enum {
    value = (sizeof(helper<T>(NULL)) == 1)
  };
};

template <bool>
struct BcastWrapper {};

template <>
struct BcastWrapper<false> {
  template <typename U>
  static void Compute(void* buf, int count, int root) {
    VLOG(V_DEBUG) << "Not Broadcasting...";
  }
};

template <>
struct BcastWrapper<true> {
  template <typename U>
  static void Compute(void* buf, int count, int root) {
    VLOG(V_DEBUG) << "Broadcasting...";
    U::Compute(buf, count, root);
  }
};

template <typename T>
void Bcast(void* buf, int count, int root) {
  BcastWrapper<HasBcast<T>::value>::template Compute<T>(buf, count, root);
}

template <typename FILLFUNCTOR, typename T, typename BCASTFUNCTOR>//fillop, dtype
inline void VariableOpImpl<FILLFUNCTOR, T, BCASTFUNCTOR>::Compute(OpContext* context) {
  if (!initialized_) {
    Tensor* out = context->Output(0);
    FILLFUNCTOR(op_def_).Compute(out->mutable_data<T>(), out->count());
    //{
      //vector<float> buf;
      //buf.resize(out->count(), 1);
      //FILE *fp = NULL;
      //if (op_def_.output(0) == "Variable0") {
        //fp = fopen("/users/shizhenx/projects/swCaffe/conv1", "rb");  
        //CHECK(fp);
        //LOG(INFO) << "v0";
      //}
      //if (op_def_.output(0) == "Variable2") {
        //fp = fopen("/users/shizhenx/projects/swCaffe/conv2", "rb");  
        //CHECK(fp);
        //LOG(INFO) << "v2";
      //}
      //if (op_def_.output(0) == "Variable4") {
        ////fp = fopen("/users/shizhenx/projects/swCaffe/fc1", "rb");  
        ////CHECK(fp);
        //CHECK(buf.size() == 500*800);
        //float scale = sqrt(3.f/800);
        //std::default_random_engine generator;
        //std::uniform_real_distribution<float> distribution(-scale, scale);
        //for (int i = 0; i < 500*800; i++)
          //buf[i] = distribution(generator);
        //LOG(INFO) << "v4";
        //checkCudaError(cudaMemcpy(out->mutable_data<float>(), buf.data(), out->count()*sizeof(float),
                                  //cudaMemcpyHostToDevice));
      //}
      //if (op_def_.output(0) == "Variable6") {
        ////fp = fopen("/users/shizhenx/projects/swCaffe/fc2", "rb");  
        ////CHECK(fp);
        ////LOG(INFO) << "v3";
        //CHECK(buf.size() == 10*500);
        //float scale = sqrt(3.f/500);
        //std::default_random_engine generator;
        //std::uniform_real_distribution<float> distribution(-scale, scale);
        //for (int i = 0; i < 10*500; i++)
          //buf[i] = distribution(generator);
        //LOG(INFO) << "v6";
        //checkCudaError(cudaMemcpy(out->mutable_data<float>(), buf.data(), out->count()*sizeof(float),
                                  //cudaMemcpyHostToDevice));
      //}
      //if (fp) {
        //CHECK(fread(buf.data(), out->count(), sizeof(float), fp));
        //LOG(INFO) << op_def_.output(0) << " CHECKING:\t"
                  //<< buf[0] << "\t" << buf[buf.size()-1];
        //fclose(fp);
        //checkCudaError(cudaMemcpy(out->mutable_data<float>(), buf.data(), out->count()*sizeof(float),
                                  //cudaMemcpyHostToDevice));
      //}
    //}
    initialized_ = true;
    if (out->device_type() == GPU) {
      Tensor cpu_buffer; 
      cpu_buffer.Rebase(::midend::GetAllocator(::midend::DeviceTypeToString(CPU)), *out);
      cpu_buffer.SyncWith(*out);
      Bcast<BCASTFUNCTOR>(cpu_buffer.mutable_data<T>(), cpu_buffer.count(), 0);
      out->SyncWith(cpu_buffer);
    }else {
      Bcast<BCASTFUNCTOR>(out->mutable_data<T>(), out->count(), 0);
    }
    //out->DebugNumerical<T>();
  }
};

template <typename FILLFUNCTOR, typename T, bool MPIEnable>//fillop, dtype
inline DDVOpImpl<FILLFUNCTOR, T, MPIEnable>::DDVOpImpl(const OpDef& def)
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
  if (MPIEnable) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    num_ /= size;
  }
}

template <typename FILLFUNCTOR, typename T, bool MPIEnable>//fillop, dtype
inline DDVOpImpl<FILLFUNCTOR, T, MPIEnable>::~DDVOpImpl() {
  if (buf_) free(buf_);
}

template <typename FILLFUNCTOR, typename T, bool MPIEnable>//fillop, dtype
void DDVOpImpl<FILLFUNCTOR, T, MPIEnable>::Compute(OpContext* context) {
  if (!buf_) {
    buf_ = (T*)malloc(num_*item_size_*sizeof(T));
    FILLFUNCTOR(op_def_).Compute(buf_, num_*item_size_);
  }
  int next_idx = (context->GetRound() % (num_/batch_));
  if (next_idx != curr_idx_) {
    //LOG(INFO) << "Next idx: " << next_idx << "\tCurr idx: " << curr_idx_;
    //LOG(INFO) << "batch: " << batch_ << "\titem_size: " << item_size_;
    //LOG(INFO) << "Next idx: " << next_idx
              //<< "\tCurr idx: " << curr_idx_
              //<< "\tRound: " << context->GetRound();
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

} //namespace backend

#endif
