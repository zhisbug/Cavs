#include "cavs/backend/op_impl.h"
#include "cavs/midend/allocator.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

/*namespace backend {*/

/*using ::midend::Allocator;*/
/*using ::midend::GetAllocator;*/
/*using ::midend::Tensor;*/

/*template <typename T>*/
/*class ProjectionOpKernel: public OpImpl {*/
 /*public:*/
  /*explicit ProjectionOpKernel(const OpDef& def);*/
  /*void Compute(OpContext* context) override;*/

 /*private:*/
  /*Allocator* alloc_;*/
  /*T* workspace;*/
  /*T* lamda;*/
/*};*/

/*} //namespace backend*/

template<typename T>
__device__ inline void Comparator(T& valA, T& valB, bool direction) {
  if ((valA > valB) == direction) {
    T tmp; 
    tmp = valA;
    valA = valB;
    valB = tmp;
  }
}

template<typename T, unsigned SHARE_SIZE_LIMIT>
__global__ BatchedMergeSort(T *inout, batch, unsigned N, bool direction) {
  __shared__ T s_val[SHARE_SIZE_LIMIT];

  T* d_val = inout + blockIdx.x*SHARE_SIZE_LIMIT + threadIdx.x;
  s_val[threadIdx.x] = d_val[0];
  s_val[threadIdx.x+(SHARE_SIZE_LIMIT)/2] = d_val[SHARE_SIZE_LIMIT/2];

  for (unsigned size = 2; size < N; size <<= 1) {
    unsigned stride = size / 2; 
    unsigned offset = threadIdx.x & (stride -1);
    __syncthreads();
    {
      unsigned pos = 2*threadIdx.x - offset; 
      Comparator(s_val[pos], s_val[pos+stride], direction);
      stride >>= 1;
    }
    for (; stride > 0; stride >>= 1) {
      __syncthreads(); 
      unsigned pos = 2*threadIdx.x - (threadIdx.x&(stride-1));
      if (offset >= stride) {
        Comparator(s_val[pos-stride], s_val[pos], direction);
      }
    }
  }
  __syncthreads();
  d_val[0] = s_val[threadIdx.x];
  d_val[SHARE_SIZE_LIMIT/2] = s_val[threadIdx.x+(SHARE_SIZE_LIMIT)/2];
}
