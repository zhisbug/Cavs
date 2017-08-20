#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/allocator.h"
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
    CHECK(y->dims() > 0);
    CHECK(y->dims(0) == -1 || x.count() >= y->count())
         << x.name() << "\t" << x.count() << "\t" << x.debug_size() << "Bytes\n"
         << y->name() << "\t" << y->count() << "\t" << y->debug_size() << "Bytes\n";
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
  CHECK(y->dims() == 2) << y->debug_info();
  CHECK(x.dims(0) == y->dims(0));
  CHECK(x.count() == y->count());
}

template <typename T>
class ReshapeLikeOp : public ReshapeOpImpl {
 public:
  explicit ReshapeLikeOp(const OpDef& def) :
    ReshapeOpImpl(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
void ReshapeLikeOp<T>::Compute(OpContext* context) {
  const Tensor& dy = context->Input(0);
  const Tensor& x  = context->Input(1);
  Tensor* dx = context->Output(0);
  //VLOG(V_DEBUG) << x.debug_info() << dy.debug_info() << dx->debug_info();
  CHECK(dx->count() == dy.count());
  CHECK(x.dims() == dx->dims());
  if (x.IsDynamicShape()) {
    CHECK(dx->IsDynamicShape());
    for (int i = 1; i < x.dims(); i++)
      CHECK(x.dims(i) == dx->dims(i));
    CHECK(x.debug_size() == dx->debug_size());
  }else {
    for (int i = 0; i < x.dims(); i++)
      CHECK(x.dims(i) == dx->dims(i));
    CHECK(x.count() == dx->count());
  }
  //checkCudaError(cudaMemcpy(dx->mutable_data<T>(), dy.data<T>(),
                            //dy.count()*sizeof(T), cudaMemcpyDeviceToDevice));
  dy.DebugNumerical<T>();
  dx->DebugNumerical<T>();
}

class ExpandDimsOpImpl : public ReshapeOpImpl {
 public:
  explicit ExpandDimsOpImpl(const OpDef& def) :
    ReshapeOpImpl(def) {
    axis_ = GetSingleArg<int>(def, "Axis");
    CHECK(axis_ >= 0);
  }

  void Compute(OpContext* context) override {
    const Tensor& x = context->Input(0);
    Tensor* y = context->Output(0);
    CHECK(x.dims() > 0);
    //we loose this constraint for the batching,
    //100==>1, 100 before batching 
    //1, 100 ==>1, 100 after batching
    CHECK(y->dims() == x.dims() + 1 ||
         (y->dims() == x.dims() && x.IsDynamicShape()));
    //for (int i = 0; i < y->dims(); i++) {
      //if (i < axis_) 
        //CHECK(x.dims(i) == y->dims(i));
      //else if (i == axis_)
        //CHECK(1 == y->dims(i));
      //else
        //CHECK(x.dims(i-1) == y->dims(i));
    //}
    x.DebugNumerical<float>();
    y->DebugNumerical<float>();
  }

 private:
  int axis_;
};

REGISTER_OP_IMPL_BUILDER(Key("Reshape").Device("GPU"), ReshapeOpImpl);
REGISTER_OP_IMPL_BUILDER(Key("Reshape").Device("CPU"), ReshapeOpImpl);
REGISTER_OP_IMPL_BUILDER(Key("Flatten").Device("GPU"), FlattenOp);
REGISTER_OP_IMPL_BUILDER(Key("Flatten").Device("CPU"), FlattenOp);
REGISTER_OP_IMPL_BUILDER(Key("Expand_dims").Device("GPU"), ExpandDimsOpImpl);
REGISTER_OP_IMPL_BUILDER(Key("Expand_dims").Device("CPU"), ExpandDimsOpImpl);
REGISTER_OP_IMPL_BUILDER(Key("ReshapeLike").Device("GPU"), ReshapeLikeOp<float>);
//REGISTER_OP_IMPL_BUILDER(Key("ReshapeLike").Device("CPU"), ReshapeLikeOp<float>);

} //namespace backend
