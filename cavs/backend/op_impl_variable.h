#ifndef CAVS_BACKEND_OP_IMPL_VARIABLE_H_
#define CAVS_BACKEND_OP_IMPL_VARIABLE_H_

#include "cavs/backend/op_impl.h"
#include "cavs/backend/op_impl_mpi_functor.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/op_context.h"
#include "cavs/midend/tensor.h"

#include <vector>

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

template <typename FILLFUNCTOR, typename T, typename BCASTFUNCTOR=bool>//fillop, dtype
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
  }
};

template <typename FILLFUNCTOR, typename T, typename BCASTFUNCTOR>//fillop, dtype
inline DDVOpImpl<FILLFUNCTOR, T, BCASTFUNCTOR>::DDVOpImpl(const OpDef& def)
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

template <typename FILLFUNCTOR, typename T, typename BCASTFUNCTOR>//fillop, dtype
inline DDVOpImpl<FILLFUNCTOR, T, BCASTFUNCTOR>::~DDVOpImpl() {
  if (buf_) free(buf_);
}

template <typename FILLFUNCTOR, typename T, typename BCASTFUNCTOR>//fillop, dtype
void DDVOpImpl<FILLFUNCTOR, T, BCASTFUNCTOR>::Compute(OpContext* context) {
  if (!buf_) {
    buf_ = (T*)malloc(num_*item_size_*sizeof(T));
    FILLFUNCTOR(op_def_).Compute(buf_, num_*item_size_);
    //MPIBcastFunctor<T>::Compute(buf_, num_*item_size_, 0);
    Bcast<BCASTFUNCTOR>(buf_, num_*item_size_, 0);
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

} //namespace backend

#endif
