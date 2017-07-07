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
  explicit GraphGatherOp(const OpDef& def) : OpImpl(def) {
    child_idx_ = GetSingleArg<int>(def, "Child");
    CHECK(child_idx_ >= 0);
    offset_ = GetSingleArg<int>(def, "Offset");
    CHECK(offset_ >= 0);
  }

  void Compute(OpContext* context) override {
    LOG(FATAL) << "Gather Operator needs further runtime support";
    int job_id = GraphScheduler::GetJobId();
    int child_id = GraphScheduler::child_id(job_id, child_idx_);
    Tensor* out = context->Output(0);
    checkCudaError(cudaMemcpy(out->mutable_data<T>(), GraphScheduler::buffer(child_id)+offset_*sizeof(T), 
                              out->count()*sizeof(T), cudaMemcpyDeviceToDevice));
  }

 private:
  int child_idx_;
  int offset_;
};

template <typename T>
class GraphScatterOp : public OpImpl {
 public:
  explicit GraphScatterOp(const OpDef& def) : OpImpl(def) {}

  void Compute(OpContext* context) override {
    LOG(FATAL) << "Push Operator needs further runtime support";
    int job_id = GraphScheduler::GetJobId();
    const Tensor& inp = context->Input(0);
    GraphScheduler::SetUnit(inp.count()*sizeof(T));
    checkCudaError(cudaMemcpy(GraphScheduler::buffer(job_id), inp.data<T>(),
                              inp.count()*sizeof(T), cudaMemcpyDeviceToDevice));
  }
};

template <typename T>
class GraphPushOp : public OpImpl {
 public:
  explicit GraphPushOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    LOG(FATAL) << "Push Operator needs further runtime support";
    int job_id = GraphScheduler::GetJobId();
    const Tensor& inp = context->Input(0);
    Tensor* out = context->Output(0);
    checkCudaError(cudaMemcpy(out->mutable_data<T>()+job_id, inp.data<T>(), 
                              inp.count()*sizeof(T), cudaMemcpyDeviceToDevice));
  }
};

template <typename T>
class GraphPullOp : public OpImpl {
 public:
  explicit GraphPullOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    LOG(FATAL) << "Pull Operator needs further runtime support";
    int job_id = GraphScheduler::GetJobId();
    const Tensor& inp = context->Input(0);
    Tensor* out = context->Output(0);
    checkCudaError(cudaMemcpy(out->mutable_data<T>(), inp.data<T>()+job_id, 
                              out->count()*sizeof(T), cudaMemcpyDeviceToDevice));
  }
};

template <typename T>
class GraphOutputOp : public OpImpl {
 public:
  explicit GraphOutputOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    //do nothing now...
    LOG(FATAL) << "graphoutput Operator needs further runtime support";
  }
};

template <typename T>
class GraphOutputGradOp : public OpImpl {
 public:
  explicit GraphOutputGradOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    //do nothing now...
    LOG(FATAL) << "graphoutputgrad Operator needs further runtime support";
  }
};

REGISTER_OP_IMPL_BUILDER(Key("GraphOutput").Device("GPU"), GraphOutputOp<float>);
REGISTER_OP_IMPL_BUILDER(Key(GetGradientName("GraphOutput")).Device("GPU"), GraphOutputGradOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Pull").Device("GPU"),    GraphPullOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Push").Device("GPU"),    GraphPushOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Scatter").Device("GPU"), GraphScatterOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Gather").Device("GPU"),  GraphGatherOp<float>);

} //namespace backend
