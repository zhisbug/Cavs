#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/allocator.h"
//#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/cudnn_types.h"

namespace backend {

using ::midend::Tensor;

class ReshapeOpImpl : public OpImpl {
 public:
  explicit ReshapeOpImpl(const OpDef& def) :
    OpImpl(def) {}
  void Compute(OpContext* context) override {
    const Tensor& x = context->Input(0);
    Tensor* y = context->Output(0);
    CHECK(x.count() == y->count());
    x.DebugNumerical<float>();
    y->DebugNumerical<float>();
  }
};

class FlattenOp : public ReshapeOpImpl {
 public:
  explicit FlattenOp(const OpDef& def) :
    ReshapeOpImpl(def) {}
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

class ReshapeLikeOp : public ReshapeOpImpl {
 public:
  explicit ReshapeLikeOp(const OpDef& def) :
    ReshapeOpImpl(def) {}
  void Compute(OpContext* context) override;
};

void ReshapeLikeOp::Compute(OpContext* context) {
  const Tensor& dy = context->Input(0);
  const Tensor& x  = context->Input(1);
  Tensor* dx = context->Output(0);
  CHECK(x.count() == dx->count());
  CHECK(x.count() == dy.count());
  CHECK(x.dims() == dx->dims());
  for (int i = 0; i < x.dims(); i++)
    CHECK(x.dims(i) == dx->dims(i));

  dy.DebugNumerical<float>();
  dx->DebugNumerical<float>();
}

REGISTER_OP_IMPL_BUILDER(Key("Reshape").Device("GPU"), ReshapeOpImpl);
REGISTER_OP_IMPL_BUILDER(Key("Reshape").Device("CPU"), ReshapeOpImpl);
REGISTER_OP_IMPL_BUILDER(Key("Flatten").Device("GPU"), FlattenOp);
REGISTER_OP_IMPL_BUILDER(Key("Flatten").Device("CPU"), FlattenOp);
REGISTER_OP_IMPL_BUILDER(Key("ReshapeLike").Device("GPU"), ReshapeLikeOp);
REGISTER_OP_IMPL_BUILDER(Key("ReshapeLike").Device("CPU"), ReshapeLikeOp);

} //namespace backend
