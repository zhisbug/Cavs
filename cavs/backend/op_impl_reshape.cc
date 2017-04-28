#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/allocator.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/cudnn_types.h"

namespace backend {

using ::midend::Tensor;

class FlattenOp: public OpImpl {
 public:
  explicit FlattenOp(const OpDef& def) :
    OpImpl(def) {}
  void Compute(OpContext* context) override;
};

void FlattenOp::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  Tensor* y = context->Output(0);
  CHECK(x.dims() == 4);
  CHECK(y->dims() == 2) << y->DebugInfo();
  CHECK(x.dims(0) == y->dims(0));
  CHECK(x.count() == y->count());
}

class ReshapeLikeOp: public OpImpl {
 public:
  explicit ReshapeLikeOp(const OpDef& def) :
    OpImpl(def) {}
  void Compute(OpContext* context) override;
};

void ReshapeLikeOp::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  Tensor* y = context->Output(0);
  CHECK(x.count() == y->count());
  //x.DebugNumerical<float>();
  //y->DebugNumerical<float>();
}

REGISTER_OP_IMPL_BUILDER(Key("Flatten").Device("GPU"), FlattenOp);
REGISTER_OP_IMPL_BUILDER(Key("Flatten").Device("CPU"), FlattenOp);
REGISTER_OP_IMPL_BUILDER(Key("ReshapeLike").Device("GPU"), ReshapeLikeOp);
REGISTER_OP_IMPL_BUILDER(Key("ReshapeLike").Device("CPU"), ReshapeLikeOp);

} //namespace backend
