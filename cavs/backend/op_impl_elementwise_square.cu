#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"

namespace backend {

using ::midend::Tensor;

template <typename T>
class SquareGradOpImpl: public OpImpl {
 public:
  explicit SquareGradOpImpl(const OpDef& def)
    : OpImpl(def) {}
  void Compute(OpContext* context) override;
};

template <typename T> 
__global__ void Kernel(T* out, const T* inp0, const T* inp1,
    const T alpha, size_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    out[i] = inp0[i]*inp1[i]*alpha; 
  } 
}

template <typename T>
void SquareGradOpImpl<T>::Compute(OpContext* context) {
  const Tensor& inp0 = context->Input(0);
  const Tensor& inp1 = context->Input(1);
  Tensor* y = context->Output(0);

  CHECK(inp0.dims() == inp1.dims());
  CHECK(inp0.dims() == y->dims());
  CHECK(inp0.count() == inp1.count());
  CHECK(inp0.count() == y->count());
  Kernel<T><<<BLOCKS_PER_GRID(inp0.count()), THREADS_PER_BLOCK>>>(
    y->mutable_data<T>(), inp0.data<T>(), inp1.data<T>(), 2, inp0.count());
  checkCudaError(cudaGetLastError());
}

REGISTER_OP_IMPL_BUILDER(Key(GetGradientName("Square")).Device("GPU"),
    SquareGradOpImpl<float>);

} //namespace backend
