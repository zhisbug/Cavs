#ifndef CAVS_BACKEND_CUBLAS_WRAPPER_H_
#define CAVS_BACKEND_CUBLAS_WRAPPER_H_

namespace backend {

//cublas only supports FORTRAN order, fuck
//so we have to wrap a new interface

//level3
template <typename T>
void MatMulMatCublasWrapper(
    const bool TransA, const bool TransB, 
    const int M, const int N, const int K, 
    const T alpha, const T* A, const T* B,
    const T beta, T* C);

//level2
template <typename T>
void MatMulVecCublasWrapper(
    const bool TransA,
    const int M, const int N,
    const T alpha, const T* A, const T* B,
    const T beta, T* C);

//level1
template <typename T>
void AxpyCublasWrapper(
    const int N, const T alpha,
    const T* x, T* y);

//level1
template <typename T>
void ScalCublasWrapper(
    const int N, const T alpha,
    T* x);

//level1
template <typename T>
void AsumCublasWrapper(
    const int N, const T* x,
    T* y);

//level1
template <typename T>
void ArgminCublasWrapper(
    const int N, const T* x,
    int* index);

//level1
//ALERT: the index is 1 to N, not 0 to N-1
template <typename T>
void ArgmaxCublasWrapper(
    const int N, const T* x,
    int* index);

} //namespace backend

#endif
