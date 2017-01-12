#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

using ::midend::Allocator;
using ::midend::GetAllocator;
using ::midend::DeviceTypeToString;
using ::midend::Tensor;

template <typename T>
class MatMulOpCublas : public OpImpl {
 public:
  explicit MatMulOpCublas(const OpDef& def);
  void Compute(OpContext* context) override;

 private:
};

} //namespace backend
