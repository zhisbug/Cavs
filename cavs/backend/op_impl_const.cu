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
__global__ void ConstKernel(T* out, const T inp, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = inp; 
  }
}

template <typename T>
class ConstOpImpl : public OpImpl {
 public:
  explicit ConstOpImpl(const OpDef& def) : OpImpl(def) {
    value = GetSingleArg<T>(op_def_, "init");
  }

  void Compute(OpContext* context) override {
    Tensor* out = context->Output(0);
    int n = out->count();
    ConstKernel<T><<<BLOCKS_PER_GRID(n), THREADS_PER_BLOCK>>> (
        out->mutable_data<T>(), value, n);
  }

 private:
  T value;
};

REGISTER_OP_IMPL_BUILDER(Key("ConstOp").Device("GPU"), ConstOpImpl<float>);

} //namespace backend
