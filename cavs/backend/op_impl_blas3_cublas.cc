#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

using ::midend::Tensor;

template <typename T>
class MatMulMatOpCublas : public OpImpl {
 public:
  explicit MatMulMatOpCublas(const OpDef& def)
    : OpImpl(def) {}
  void Compute(OpContext* context) override;

 private:
};

template <typename T>
void MatMulMatOpCublas<T>::Compute(OpContext* context) {
  const Tensor& A = context->Input(0);
  const Tensor& B = context->Input(1);
  Tensor* C = context->Output(0);

  int M = A.dims(0);
  int K = A.dims(1);
  CHECK(B.dims(0) == K);
  int N = B.dims(1);

  MatMulMatCublasWrapper<T>(false, false,
      M, N, K, 1.f, A.data<T>(), B.data<T>(),
      0, C->mutable_data<T>());
}

} //namespace backend
