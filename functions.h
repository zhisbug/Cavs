#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_
#include <stdio.h>

template <typename Dtype>
void cublasWrapper(const bool TransA, const bool TransB, 
        const int M, const int N, const int K, 
        const Dtype alpha, const Dtype *A, const Dtype *B,
        const Dtype beta, Dtype *C);
template <typename Dtype>
void atlasWrapper(const bool TransA, const bool TransB, 
        const int M, const int N, const int K, 
        const Dtype alpha, const Dtype *A, const Dtype *B,
        const Dtype beta, Dtype *C);
template <typename Dtype>
void eigenWrapper(const bool TransA, const bool TransB, 
        const int M, const int N, const int K, 
        const Dtype alpha, const Dtype *A, const Dtype *B,
        const Dtype beta, Dtype *C);
template <typename Dtype>
void MKLWrapper(const bool TransA, const bool TransB, 
        const int M, const int N, const int K, 
        const Dtype alpha, const Dtype *A, const Dtype *B,
        const Dtype beta, Dtype *C);

template <typename Dtype>
class Tensor{
public:
    Tensor(int count);
    Dtype *cpu_buf(){return (Dtype*)cpu_buf_;}
    Dtype *gpu_buf(){return (Dtype*)gpu_buf_;}
    void sync2d();
    void sync2h();
private:
    void *cpu_buf_;
    void *gpu_buf_;
    size_t capacity_;
};
#endif
