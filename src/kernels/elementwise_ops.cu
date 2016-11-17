#include "elementwise_ops.h"

template <typename OP, typename T, typename R = T> 
__global__ void UnaryKernel(R* out, const T* inp, int n) {
    CUDA_1D_KERNEL_LOOP(i, n) { 
        out[i] = OP()(inp[i]); 
    } 
}

template <typename OP, typename T, typename R = T> 
__global__ void BinaryKernel(R* out, const T* inp0, const T* inp1, int n) {
    CUDA_1D_KERNEL_LOOP(i, n) { 
        out[i] = OP()(inp0[i], inp1[i]); 
    } 
}

template <typename OP, typename T, typename R = T>
struct CUDAUnaryFunctor: UnaryFunctor<T> {
    void operator () (R* out, const T* inp, int n) override {
        UnaryKernel<OP, T><<<THREADS_PER_BLOCK, BLOCK_PER_GRID(n)>>>(
            out, inp, n);
    }
}

template <typename OP, typename T, typename R = T>
struct CUDABinaryFunctor: BinaryFunctor<T> {
    void operator () (R* out, const T* inp0, const T* inp1, int n) override {
        BinaryKernel<OP, T><<<THREADS_PER_BLOCK, BLOCK_PER_GRID(n)>>>(
            out, inp0, inp1, n);
    }
}

REGISTER_OP_BUILDER(Key("Abs").Device("GPU"), CUDAUnaryFunctor<Math::Abs, float>);
REGISTER_OP_BUILDER(Key("Add").Device("GPU"), CUDABinaryFunctor<Math::Add, float>);
