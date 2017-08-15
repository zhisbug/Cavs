#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/stream_event_handle_pool.h"

namespace backend {

using ::midend::Tensor;

template <typename T>
class MatMulMatOpCublas : public OpImpl {
 public:
  explicit MatMulMatOpCublas(const OpDef& def);
  void Compute(OpContext* context) override;

 private:
  bool TransA;
  bool TransB;
  cublasHandle_t handle_;
};

template <typename T>
MatMulMatOpCublas<T>::MatMulMatOpCublas(const OpDef& def)
    : OpImpl(def), TransA(false), TransB(false), handle_(NULL) {
  for (auto& t : GetListArg<int>(op_def_, "Transpose")) {
    LOG(INFO) << "Transpose: " << t;
    if (t == 0) TransA = true;
    if (t == 1) TransB = true;
  }
}

template <typename T>
void MatMulMatOpCublas<T>::Compute(OpContext* context) {
  const Tensor& A = context->Input(0);
  const Tensor& B = context->Input(1);
  Tensor* C = context->Output(0);

  int MA = (TransA == false)? A.dims(0) : A.dims(1);
  int KA = (TransA == false)? A.dims(1) : A.dims(0);
  int KB = (TransB == false)? B.dims(0) : B.dims(1);
  int NB = (TransB == false)? B.dims(1) : B.dims(0);
  CHECK(KA == KB);
  CHECK(C->dims(0) == MA)
    << "C.dims(0): " << C->dims(0)
    << "\tMA: "      << MA;
  CHECK(C->dims(1) == NB)
    << "C.dims(1): " << C->dims(1)
    << "\tNB: "      << NB;

  //VLOG(V_DEBUG) << "here";
  if (!handle_) {
    if (context->GetStreamID() != -1) {
      handle_ = StreamEventHandlePool::GetCublasHandle(context->GetStreamID());
      VLOG(V_DEBUG) << "running with stream_id: " << context->GetStreamID();
    }else {
      handle_ = CudaCommon::cublasHandle();
      VLOG(V_DEBUG) << "running with NULL stream";
    }
  }
  //LOG(FATAL) << "here";

  MatMulMatCublasWrapper<T>(handle_, TransA, TransB,
      MA, NB, KA, 1.f, A.data<T>(), B.data<T>(),
      0, C->mutable_data<T>());
  A.DebugNumerical<T>();
  B.DebugNumerical<T>();
  C->DebugNumerical<T>();
}

REGISTER_OP_IMPL_BUILDER(Key("MatMul").Device("GPU"), MatMulMatOpCublas<float>);

} //namespace backend
