#ifndef CAVS_BACKEND_OP_IMPL_VARIABLE_H_
#define CAVS_BACKEND_OP_IMPL_VARIABLE_H_

#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/op_context.h"
#include "cavs/proto/tensor_shape.pb.h"

#include <vector>

using std::vector;

namespace backend {

using ::midend::OpContext;
using ::midend::Tensor;

template <typename FILLFUNCTOR, typename T>//fillop, dtype
class VariableOpImpl : public OpImpl {
 public:
  explicit VariableOpImpl(const OpDef& def)
    : OpImpl(def), initialized_(false) {
  }

  void Compute(OpContext* context) override {
    if (!initialized_) {
      Tensor* out = context->Output(0);
      if (out->dims(0) == 5000) {
        vector<float> buf(5000*100);
        FILE *fp = fopen("/users/shizhenx/projects/tm_final/code/doc_tpc.init", "rb");
        CHECK(fp);
        fread(buf.data(), sizeof(float), 5000*100, fp);
        checkCudaError(cudaMemcpy(out->mutable_data<T>(), buf.data(), 5000*100*sizeof(float),
                                  cudaMemcpyHostToDevice));
        fclose(fp);
        LOG(INFO) << "doc_tpc.init...";
        //out->DebugNumerical<float>();
        initialized_ = true;
        return;
      }else if (out->dims(0) == 100) {
        vector<float> buf(100*1000);
        FILE *fp = fopen("/users/shizhenx/projects/tm_final/code/tpc_word.init", "rb");
        CHECK(fp);
        fread(buf.data(), sizeof(float), 100*1000, fp);

        checkCudaError(cudaMemcpy(out->mutable_data<T>(), buf.data(), 100*1000*sizeof(float),
                                  cudaMemcpyHostToDevice));
        fclose(fp);
        LOG(INFO) << "tpc_word.init...";
        out->DebugNumerical<float>();
        initialized_ = true;
        return;
      }
      FILLFUNCTOR(op_def_).Compute(out->mutable_data<T>(), out->count());
      out->DebugNumerical<T>();
      initialized_ = true;
    }
  }
 private:
  bool initialized_;
};

} //namespace backend

#endif
