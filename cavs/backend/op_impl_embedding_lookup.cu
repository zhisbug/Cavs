#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
/*#include "cavs/midend/devices.h"*/
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

using ::midend::Tensor;

template <typename T>
class EmbeddingLookupOp: public OpImpl {
 public:
  explicit EmbeddingLookupOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
__global__ void BatchedCopy(T *embedding,
    const T* data, const T* matrix,
    int embedding_size) {
  int output_offset = blockIdx.x*embedding_size;
  int matrix_offset = data[blockIdx.x]*embedding_size;
  for (int round = 0; round < (embedding_size+blockDim.x-1)/blockDim.x; round++) {
    int offset_within_vec = threadIdx.x + round*blockDim.x;
    if (offset_within_vec < embedding_size) {  
      embedding[output_offset+offset_within_vec] =
        matrix[matrix_offset+offset_within_vec];
    }
  }
}

template <typename T>
void EmbeddingLookupOp<T>::Compute(OpContext* context) {
  const Tensor& input = context->Input(0);
  const Tensor& embedding_matrix = context->Input(1);
  Tensor* embedding = context->Output(0);

  CHECK(embedding_matrix.dims() == 2);
  int vocabulary_size = embedding_matrix.dims(0);
  int embedding_size  = embedding_matrix.dims(1);
  CHECK(vocabulary_size >= embedding_size);
  CHECK(embedding->dims() == input.dims()+1);
  for (int i = 0; i < input.dims(); i++)
    CHECK(embedding->dims(i) == input.dims(i));
  CHECK(embedding->dims(embedding->dims()-1) == embedding_size);

  int slices = input.count();
  const int MAX_THREADS_IN_BLOCK = 1 << 10;
  int threadsPerBlock = (MAX_THREADS_IN_BLOCK > embedding_size) ?
                         embedding_size : MAX_THREADS_IN_BLOCK;
  int blocksPerGrid = slices;
  BatchedCopy<<<blocksPerGrid, threadsPerBlock>>>(
      embedding->mutable_data<T>(),
      input.data<T>(), embedding_matrix.data<T>(),
      embedding_size);

  input.DebugNumerical<T>();
  embedding_matrix.DebugNumerical<T>();
  embedding->DebugNumerical<T>();
}

template <typename T>
class EmbeddingLookupGradOp: public OpImpl {
 public:
  explicit EmbeddingLookupGradOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override;
};

template <typename T>
__global__ void BatchedSparseUpdate(T *dMatrix,
    const T* data, const T* dY,
    int embedding_size) {
  int dY_offset = blockIdx.x*embedding_size;
  int dMatrix_offset = data[blockIdx.x]*embedding_size;
  for (int round = 0; round < (embedding_size+blockDim.x-1)/blockDim.x; round++) {
    int offset_within_vec = threadIdx.x + round*blockDim.x;
    if (offset_within_vec < embedding_size) {  
      atomicAdd(&(dMatrix[dMatrix_offset+offset_within_vec]), 
        dY[dY_offset+offset_within_vec]);
    }
  }
}

template <typename T>
void EmbeddingLookupGradOp<T>::Compute(OpContext* context) {
  const Tensor& dY = context->Input(0);
  const Tensor& input = context->Input(1);
  Tensor* dMatrix= context->Output(0);
  //we don't calculate the dX, because dX is not passed backward

  CHECK(dMatrix->dims() == 2);
  int vocabulary_size = dMatrix->dims(0);
  int embedding_size  = dMatrix->dims(1);
  CHECK(vocabulary_size >= embedding_size);
  CHECK(dY.dims() == input.dims()+1);
  for (int i = 0; i < input.dims(); i++)
    CHECK(dY.dims(i) == input.dims(i));
  CHECK(dY.dims(dY.dims()-1) == embedding_size);

  checkCudaError(cudaMemset(dMatrix->mutable_data<T>(), 0, 
                            dMatrix->count()*sizeof(T)));
  int slices = input.count();
  const int MAX_THREADS_IN_BLOCK = 1 << 10;
  int threadsPerBlock = (MAX_THREADS_IN_BLOCK > embedding_size) ?
                         embedding_size : MAX_THREADS_IN_BLOCK;
  int blocksPerGrid = slices;
  BatchedSparseUpdate<<<blocksPerGrid, threadsPerBlock>>>(
      dMatrix->mutable_data<T>(),
      input.data<T>(), dY.data<T>(),
      embedding_size);

  //A.DebugNumerical<T>();
  //B.DebugNumerical<T>();
  //C->DebugNumerical<T>();
}

REGISTER_OP_IMPL_BUILDER(Key("EmbeddingLookup").Device("GPU"), EmbeddingLookupOp<float>);
REGISTER_OP_IMPL_BUILDER(Key(GetGradientName("EmbeddingLookup")).Device("GPU"), EmbeddingLookupGradOp<float>);

} //namespace backend

