#include "functions.h"
#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_bool(gpu, true, "whether to use GPU");
DEFINE_int32(lib, 0, "0: atlas, 1:eigen, 2:MKL");
DEFINE_int32(mb, 100, "mini_batch size");
DEFINE_int32(M, 99, "mini_batch size");
DEFINE_int32(N, 100, "mini_batch size");
DEFINE_int32(K, 100, "mini_batch size");

template <typename Dtype>
void rawGemm(const int M, const int N, const int K,
        const Dtype *A, const Dtype *B, Dtype *C){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Dtype sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j]; 
            }
            C[i*N + j] = sum;
        }
    }
}

template <typename Dtype>
void check(const int M, const int N, const Dtype *C, const Dtype *C_test){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i*N + j] != C_test[i*N + j])
                LOG(FATAL)  << "i:" << i 
                            << "\tj:" << j 
                            << "\t" << C[i*N + j] 
                            << "\t" << C_test[i*N + j];
        }
    }
    LOG(INFO) << "Test passed";
}

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    Tensor<float> A(FLAGS_M*FLAGS_K);
    Tensor<float> B(FLAGS_K*FLAGS_N);
    Tensor<float> C(FLAGS_M*FLAGS_N);
    Tensor<float> C_test(FLAGS_M*FLAGS_N);
    for (int i = 0; i < FLAGS_M; i++) {
        for (int j = 0; j < FLAGS_K; j++) {
            A.cpu_buf()[i*FLAGS_K+j] = 1.f;
        }
    }
    for (int i = 0; i < FLAGS_K; i++) {
        for (int j = 0; j < FLAGS_N; j++) {
            B.cpu_buf()[i*FLAGS_N+j] = 1.f;
        }
    }
    if (FLAGS_gpu) {
        A.sync2d(); B.sync2d();
        cublasWrapper(false, false, FLAGS_M, FLAGS_N, FLAGS_K,
                1.f, A.gpu_buf(), B.gpu_buf(), 0.f, C.gpu_buf());
        C.sync2h();
    }else{
        if (FLAGS_lib == 0) {
             
        }else if (FLAGS_lib == 1){
        
        }else if (FLAGS_lib == 2){
        
        }
    }
    rawGemm(FLAGS_M, FLAGS_N, FLAGS_K, A.cpu_buf(), B.cpu_buf(), C_test.cpu_buf());
    check(FLAGS_M, FLAGS_N, C.cpu_buf(), C_test.cpu_buf());
    return 0;
}


