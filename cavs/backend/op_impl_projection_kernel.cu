#include "cavs/backend/op_impl.h"
#include "cavs/backend/functor_sort_scan.cuh"
#include "cavs/backend/functor_elementwise.h"
#include "cavs/midend/allocator.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/mpi_types.h"

#include <vector>
#include <iomanip>

using std::vector;

namespace backend {

using ::midend::Allocator;
using ::midend::GetAllocator;
using ::midend::DeviceTypeToString;
using ::midend::Tensor;

template <typename T>
class ProjectionOpKernel: public OpImpl {
 public:
  explicit ProjectionOpKernel(const OpDef& def);
  ~ProjectionOpKernel();
  void Compute(OpContext* context) override;

 private:
  Allocator* alloc_;
  T* workspace_sort;
  T* workspace_scan;
  T* lamda;
};

template <typename T>
ProjectionOpKernel<T>::ProjectionOpKernel(const OpDef& def)
    : OpImpl(def), workspace_sort(NULL), workspace_scan(NULL), lamda(NULL) {
  alloc_ = GetAllocator(DeviceTypeToString(GPU));
}

template <typename T>
ProjectionOpKernel<T>::~ProjectionOpKernel() {
  if (lamda)
    alloc_->Deallocate<char>((char*)lamda); 
  if (workspace_sort)
    alloc_->Deallocate<char>((char*)workspace_sort); 
  if (workspace_scan)
    alloc_->Deallocate<char>((char*)workspace_scan); 
}

template <typename T>
__inline__ __device__
T warpReduceMax(T val) {
  const int warpSize = 32;
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    val = math::Max<T>::Compute(__shfl_down(val, offset), val);
  }
  return val;
}

template <typename T>
__global__ void BatchedFindMax(T *out, const T* mu, const T* mu_scan, int N) {
  extern __shared__ int index_buf[];
  const int warpSize = 32;
  int lane = threadIdx.x%warpSize;
  int warp_id = threadIdx.x/warpSize;
  int offset = blockIdx.x*N;
  int max_index = 0;
  for (int round = 0; round < (N+blockDim.x-1)/blockDim.x; round++) {
    int offset_within_vec = threadIdx.x + round*blockDim.x;
    int idx = offset + offset_within_vec;
    if (offset_within_vec < N) {  
      if ((mu[idx] + (1.f - mu_scan[idx])/(idx+1)) > 0) {
        max_index = offset_within_vec;
      }
    }
  }

  max_index = warpReduceMax(max_index);
  if (lane == 0) 
    index_buf[warp_id] = max_index;
  __syncthreads();
  max_index = (threadIdx.x < blockDim.x / warpSize) ? index_buf[lane] : 0;
  if (warp_id == 0)
    max_index = warpReduceMax(max_index);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = (1-mu_scan[max_index])/(max_index+1);
  }
}

template <typename T>
__global__ void BatchedGetOutput(T *x, const T* y, const T* lamda, int N) {
  int offset = blockIdx.x*N;
  for (int round = 0; round < (N+blockDim.x-1)/blockDim.x; round++) {
    int offset_within_vec = threadIdx.x + round*blockDim.x;
    int idx = offset + offset_within_vec;
    if (offset_within_vec < N) {  
      T tmp = y[idx] + lamda[blockIdx.x];
      if (tmp > 0)
        x[idx] = tmp;
      else
        x[idx] = 0;
    }
  }
}

template <typename T>
void ProjectionOpKernel<T>::Compute(OpContext* context) {
  const Tensor& var_in = context->Input(0);
  Tensor* var_out = context->Output(0);
  CHECK(var_in.dims(0) == var_out->dims(0));
  CHECK(var_in.count() == var_out->count());

  int batch = var_in.dims(0);
  int N = var_in.count()/var_in.dims(0);

  /*var_in.DebugNumerical<T>();*/
  //To further reduce the workspace of tpc_word, 
  //we need to split the N dimension
  const int MINI_BATCH = (batch < 1000) ? batch : 1000;
  if (!lamda) {
    /*lamda = alloc_->Allocate<T>(var_in.dims(0));*/
    lamda = alloc_->Allocate<T>(MINI_BATCH);
  }
  if (!workspace_sort) {
    /*workspace_sort = alloc_->Allocate<T>(var_in.count());*/
    workspace_sort = alloc_->Allocate<T>(MINI_BATCH*N);
  }
  if (!workspace_scan) {
    /*workspace_scan = alloc_->Allocate<T>(var_in.count());*/
    workspace_scan = alloc_->Allocate<T>(MINI_BATCH*N);
  }
  for (int offset = 0; offset < batch; offset += MINI_BATCH) {
    int curr_batch_size = (offset + MINI_BATCH > batch) ?
                          (batch - offset) : MINI_BATCH;
    {
      /*BatchedOddEvenSort(*/
          /*workspace_sort, var_in.data<T>(), false, N, batch);*/
      BatchedOddEvenSort(
          workspace_sort, var_in.data<T>()+offset*N, false, N, curr_batch_size);
    }

    {
      const int MAX_THREADS_IN_BLOCK = 1 << 10;
      int threadsPerBlock = (MAX_THREADS_IN_BLOCK > N) ? N : MAX_THREADS_IN_BLOCK;
      /*int blocksPerGrid = batch;*/
      int blocksPerGrid = curr_batch_size;
      /*BatchedScan(workspace_scan, workspace_sort, N, batch);*/
      BatchedScan(workspace_scan, workspace_sort, N, curr_batch_size);

      BatchedFindMax<T><<<blocksPerGrid, threadsPerBlock,
                          (threadsPerBlock+31)/32*sizeof(int)>>>(
          lamda, workspace_sort, workspace_scan, N);
      /*[>BatchedGetOutput<T><<<blocksPerGrid, threadsPerBlock>>>(<]*/
          /*[>var_out->mutable_data<T>(),<]*/
          /*[>var_in.data<T>(),<]*/
          /*[>lamda, N);<]*/
      BatchedGetOutput<T><<<blocksPerGrid, threadsPerBlock>>>(
          var_out->mutable_data<T>() + offset*N,
          var_in.data<T>() + offset*N,
          lamda, N);
    }
  }

  /*var_out->DebugNumerical<T>();*/
}

REGISTER_OP_IMPL_BUILDER(Key("Simplex").Device("GPU"), ProjectionOpKernel<float>);

} //namespace backend
