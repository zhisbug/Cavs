#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cudaRTC_wrapper.h"

#include <string>
#include <vector>
#include <algorithm>

namespace backend {

using ::midend::Tensor;
using std::string;
using std::vector;

template <typename T>
class FusedKernelOpImpl : public OpImpl {
 public:
  explicit FusedKernelOpImpl(const OpDef& def) : OpImpl(def) {
    const string& kernel_name = GetSingleArg<string>(def, "KernelName"); 
    const string& kernel_src  = GetSingleArg<string>(def, "KernelSource"); 
    wrapper_.Compile(kernel_name, kernel_src);
  }

  void Compute(OpContext* context) override;

 private:
  RTC::CudaRTCWrapper wrapper_;
};

template <typename T>
void FusedKernelOpImpl<T>::Compute(OpContext* context) {
  vector<void*> outputs;
  vector<void*> inputs;
  const int num_elements = context->Input(0).count();
  for (int i = 0; i < context->OutputSize(); i++) {
    outputs.push_back((void*)(context->Output(i)->mutable_data<T>())); 
    CHECK(context->Output(i)->count() == num_elements);
  }
  for (int i = 0; i < context->InputSize(); i++) {
    inputs.push_back((void*)context->Input(i).data<T>()); 
    CHECK(context->Input(i).count() == num_elements);
  }
  wrapper_.Launch(outputs, inputs, num_elements, 
      BLOCKS_PER_GRID(num_elements), 1, 1,
      THREADS_PER_BLOCK, 1, 1);
  for (int i = 0; i < context->InputSize(); i++) {
    context->Input(i).DebugNumerical<T>();
  }
  for (int i = 0; i < context->OutputSize(); i++) {
    context->Output(i)->DebugNumerical<T>();
  }
}

REGISTER_OP_IMPL_BUILDER(Key("FusedKernel").Device("GPU"), FusedKernelOpImpl<float>);

} //namespace backend
