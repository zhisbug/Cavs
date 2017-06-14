#include "cavs/backend/functor_elementwise.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/op_impl.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/op_context.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

using ::midend::OpContext;
using ::midend::Tensor;

template <typename T> 
__global__ void SGDKernel(T* out, const T* inp0, const T* inp1,
    const float lr,size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = inp0[i] - lr*inp1[i]; 
  } 
}

template <typename T>
class SGDOpImpl : public OpImpl {
 public:
  explicit SGDOpImpl(const OpDef& def)
    : OpImpl(def), lr_(0.f) {
      lr_ = GetSingleArg<float>(def, "Learning_rate");
      VLOG(V_DEBUG) << "learning_rate = " << lr_;
      /*count_ = 0;*/
  }

  void Compute(OpContext* context) override {
    /*if (count_++ > 2293*30 && count_ % 2293 == 0) {*/
        /*lr_ /= 2;*/
        /*LOG(INFO) << "Lr Changed to " << lr_;*/
    /*}*/
    const Tensor& inp0 = context->Input(0);
    const Tensor& inp1 = context->Input(1);
    inp0.DebugNumerical<T>();
    inp1.DebugNumerical<T>();
    Tensor* out = context->Output(0);
    int n = out->count();
    SGDKernel<T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>> (
        out->mutable_data<T>(),
        inp0.data<T>(), inp1.data<T>(), lr_, n);
    out->DebugNumerical<T>();
  }

 private:
  float lr_;
  /*int count_;*/
};

REGISTER_OP_IMPL_BUILDER(Key("SGD").Device("GPU"), SGDOpImpl<float>);

} //namespace backend
