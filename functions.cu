#include "functions.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

#define checkCublasError(status)                             \
        do {                                                 \
            if (status != CUBLAS_STATUS_SUCCESS) {           \
                LOG(FATAL) << "CUDA failure: "               \
                           << status;                        \
            }                                                \
        }while(0)
#define checkCudaError(status)                               \
        do {                                                 \
            if (status != cudaSuccess) {                     \
                LOG(FATAL) << "CUDA failure: "               \
                           << cudaGetErrorString(status);    \
            }                                                \
       }while(0)

class Common{
public:
    inline static cublasHandle_t cublasHandle(){return Get()->cublasHandle_;}
private:
    Common();
    static Common* Get() {static Common c; return &c;}
    cublasHandle_t cublasHandle_;
};

Common::Common(){
    checkCublasError(cublasCreate(&cublasHandle_));
}

template<>
Tensor<float>::Tensor(int c){
    capacity_ = c*sizeof(float);
    cpu_buf_ = malloc(capacity_);
    checkCudaError(cudaMalloc(&gpu_buf_, capacity_));
}

template<>
Tensor<double>::Tensor(int c){
    capacity_ = c*sizeof(double);
    cpu_buf_ = malloc(capacity_);
    checkCudaError(cudaMalloc(&gpu_buf_, capacity_));
}

template<>
void Tensor<float>::sync2d(){
    checkCudaError(cudaMemcpy(gpu_buf_, cpu_buf_, capacity_, cudaMemcpyHostToDevice));
}

template<>
void Tensor<double>::sync2d(){
    checkCudaError(cudaMemcpy(gpu_buf_, cpu_buf_, capacity_, cudaMemcpyHostToDevice));
}

template<>
void Tensor<float>::sync2h(){
    checkCudaError(cudaMemcpy(cpu_buf_, gpu_buf_, capacity_, cudaMemcpyDeviceToHost));
}

template<>
void Tensor<double>::sync2h(){
    checkCudaError(cudaMemcpy(cpu_buf_, gpu_buf_, capacity_, cudaMemcpyDeviceToHost));
}

template <>
void cublasWrapper<float>(const bool TransA, const bool TransB, 
        const int M, const int N, const int K, 
        const float alpha, const float *A, const float *B,
        const float beta, float *C){
    int lda = (TransA == false) ? K : M;
    int ldb = (TransB == false) ? N : K;
    cublasOperation_t cuTransA =
        (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;

    checkCublasError(cublasSgemm(Common::cublasHandle(), cuTransB, cuTransA,
        N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void cublasWrapper<double>(const bool TransA, const bool TransB, 
        const int M, const int N, const int K, 
        const double alpha, const double *A, const double *B,
        const double beta, double *C){
    int lda = (TransA == false) ? K : M;
    int ldb = (TransB == false) ? N : K;
    cublasOperation_t cuTransA =
        (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
    checkCublasError(cublasDgemm(Common::cublasHandle(), cuTransB, cuTransA,
        N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

