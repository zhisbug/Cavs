#include "cavs/backend/op_impl.h"
#include "cavs/midend/graph_scheduler.h"
#include "cavs/midend/tensor.h"
#include "cavs/util/macros_gpu.h"

using ::midend::Tensor;
using ::midend::GraphScheduler;

namespace backend {

template <typename T>
class GraphGatherOp : public OpImpl {
 public:
  explicit GraphGatherOp(const OpDef& def) {
    child_idx_ = GetSingleArg<int>(def, "Child");
    CHECK(child_idx_ >= 0);
    offset_ = GetSingleArg<int>(def, "Offset");
    CHECK(offset_ >= 0);
  }
  ~GraphGatherOp(); 

  void Compute(OpContext* context) override;

 private:
  int child_idx_;
  int offset_;
};

template <typename T>
void GraphGatherOp<T>::Compute(OpContext* context) {
  int parent_id = GraphScheduler::GetJobId();
  int child_id = GraphScheduler::child_id(parent_id, child_idx_);
  Tensor* out = context->Output(0);
  //out->CopyFrom(GraphScheduler::buffer(child_id)+offset*sizeof(T));
  checkCudaError(cudaMemcpy(out->mutable_data<T>(), GraphScheduler::buffer(child_id)+offset_*sizeof(T), 
                            cudaMemcpyDeviceToDevice));
}

template <typename T>
class GraphScatterOp : public OpImpl {
 public:
  explicit GraphScatterOp(const OpDef& def) {
  }
  ~GraphScatterOp(); 

  void Compute(OpContext* context) override;

 private:
};

template <typename T>
void GraphScatterOp<T>::Compute(OpContext* context) {
}

} //namespace backend
