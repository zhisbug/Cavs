#include "cavs/backend/cuda_common.h"

namespace backend {

CudaCommon::CudaCommon() {
  checkCublasError(cublasCreate(&cublasHandle_));
  checkCUDNNError(cudnnCreate(&cudnnHandle_));
}

} //namespace backend
