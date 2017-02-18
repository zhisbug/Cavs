#include "cavs/backend/functors_elementwise.h"
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
__global__ void SGDKernel(T* out, const T* inp0, const T* inp1, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = inp0[i] + inp1[i]; 
  } 
}

template <typename T>
class SGDOpImpl : public OpImpl {
 public:
  explicit SGDOpImpl(const OpDef& def)
    : OpImpl(def) {}

  void Compute(OpContext* context) override {
    const Tensor& inp0 = context->Input(0);
    const Tensor& inp1 = context->Input(1);
    Tensor* out = context->Output(0);
    int n = out->count();
    SGDKernel<T><<<THREADS_PER_BLOCK, BLOCKS_PER_GRID(n)>>> (
        out->mutable_data<T>(),
        inp0.data<T>(), inp1.data<T>(), n);
  }

 private:
  T value;
};

REGISTER_OP_IMPL_BUILDER(Key("SGD").Device("GPU"), SGDOpImpl<float>);

} //namespace backend
