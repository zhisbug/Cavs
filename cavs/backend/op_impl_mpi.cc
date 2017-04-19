#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/midend/allocator.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/mpi_types.h"
#include "cavs/util/op_util.h"

#define checkMPIError(stmt)                            \
  do {                                                 \
    char err_buffer[MPI_MAX_ERROR_STRING];             \
    int result_len;                                    \
    int ierr = (stmt);                                 \
    if (ierr != MPI_SUCCESS) {                         \
      MPI_Error_string(ierr, err_buffer, &result_len); \
      LOG(INFO) << err_buffer;                         \
      MPI_Finalize();                                  \
    }                                                  \
  }while(0) 

namespace backend {

using ::midend::Tensor;

//MPI is currently run on CPU and the communication is global
//CUDA-aware MPI will be supported later
//For MPI_Allreduce operator, only MPI_SUM is supported.
template <typename T>
class MPIAllReduceOpImpl: public OpImpl {
 public:
  explicit MPIAllReduceOpImpl(const OpDef& def)
    : OpImpl(def) {}
  void Compute(OpContext* context) override;
};

template<typename T>
void MPIAllReduceOpImpl<T>::Compute(OpContext* context) {
  const Tensor& inp = context->Input(0);
  Tensor* out = context->Output(0);
  //currently, we assume this
  LOG(INFO) << "here";
  CHECK(inp.device_type() == out->device_type());
  if (inp.device_type() != CPU) {
    Tensor cpu_buffer; 
    cpu_buffer.Rebase(::midend::GetAllocator(::midend::DeviceTypeToString(CPU)), inp);
    cpu_buffer.SyncWith(inp);
    cpu_buffer.DebugNumerical<T>();
    checkMPIError(MPI_Allreduce(MPI_IN_PLACE,
          cpu_buffer.mutable_data<T>(),
          cpu_buffer.count(), DataTypeToMPIType<T>::value,
          MPI_SUM, MPI_COMM_WORLD));
    out->SyncWith(cpu_buffer);
  }else {
    if (reinterpret_cast<int64_t>(inp.data<T>()) == 
        reinterpret_cast<int64_t>(out->mutable_data<T>())) {
      checkMPIError(MPI_Allreduce(MPI_IN_PLACE, out->mutable_data<T>(),
            inp.count(), DataTypeToMPIType<T>::value,
            MPI_SUM, MPI_COMM_WORLD));
    }else {
      checkMPIError(MPI_Allreduce(inp.data<T>(), out->mutable_data<T>(),
            inp.count(), DataTypeToMPIType<T>::value,
            MPI_SUM, MPI_COMM_WORLD));
    }
  }
}

REGISTER_OP_IMPL_BUILDER(Key("MPIAllReduce").Device("CPU"), MPIAllReduceOpImpl<float>);

} //namespace backend
