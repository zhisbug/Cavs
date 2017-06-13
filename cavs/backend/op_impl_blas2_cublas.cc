#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
//#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

using ::midend::Tensor;

template <typename T>
class MatMulVecOpCublas : public OpImpl {
 public:
  explicit MatMulVecOpCublas(const OpDef& def)
    : OpImpl(def) {}
  void Compute(OpContext* context) override;

 private:
};

template <typename T>
void MatMulVecOpCublas<T>::Compute(OpContext* context) {
  const Tensor& A = context->Input(0);
  const Tensor& x = context->Input(1);
  Tensor* y = context->Output(0);

  int M = A.dims(0);
  int N = A.dims(1);
  CHECK(x.dims(0) == N);

  MatMulVecCublasWrapper<T>(false,
      M, N, 1.f, A.data<T>(), x.data<T>(),
      0, y->mutable_data<T>());
}

} //namespace backend
