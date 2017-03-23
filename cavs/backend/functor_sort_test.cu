#include "cavs/backend/functor_sort_scan.cuh"
#include "cavs/util/logging.h"
#include "cavs/util/macros_gpu.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <iostream>

using namespace backend;

const int MAX_THREADS_IN_BLOCK = 1 << 10;
const bool direction = true; //small left 
const bool ENFORCE = false;

int main() {
  //8k documents
  for (int Batch = 1; Batch < 1 << 10; Batch <<= 1) {
    /*for (int N = 3; N <= 1 << 10; N <<= 1) {*/
    for (int N = 3; N <= 1 << 18; N <<= 1) {
      thrust::host_vector<int> h_vec(Batch*N);
      thrust::generate(h_vec.begin(), h_vec.end(), rand);
      thrust::device_vector<int> d_vec = h_vec;
      thrust::host_vector<int> h_vec_verify = h_vec;
      thrust::device_vector<int> d_vec_verify = h_vec;
      LOG(INFO) << "Testing with N = " << N
                << "\tand Batch = " << Batch << "\t...";
      if (N <= 2*MAX_THREADS_IN_BLOCK && ENFORCE) {
        //it is assumed in in-cache implementation
        int threadsPerBlock = 1;
        while (threadsPerBlock < N) { threadsPerBlock <<= 1; }
        threadsPerBlock >>= 1;
        int blocksPerGrid = Batch;
        BatchedOddEvenSortInCache<int><<<blocksPerGrid, threadsPerBlock,
          threadsPerBlock*2*sizeof(int)>>>(
            thrust::raw_pointer_cast(d_vec.data()),
            thrust::raw_pointer_cast(d_vec.data()),
            direction, N);
        checkCudaError(cudaGetLastError());
      }else {
        BatchedOddEvenSort<int>(
            thrust::raw_pointer_cast(d_vec.data()),
            thrust::raw_pointer_cast(d_vec.data()),
            direction, N, Batch);
      }
      thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

      for (int i = 0; i < Batch; i++)
        thrust::sort(d_vec_verify.begin()+i*N, d_vec_verify.begin()+(i+1)*N);

      thrust::copy(d_vec_verify.begin(), d_vec_verify.end(), h_vec_verify.begin());

      for (int i = 0; i < Batch; i++) {
        for (int j = 0; j < N; j++) {
          CHECK(h_vec[i*N+j] == h_vec_verify[i*N+j])
               << "h_vec[" << i << "][" << j << "]: "
               << h_vec[i*N+j]
               << "\th_vec_verify[" << i << "][" << j << "]: "
               << h_vec_verify[i*N+j];
        }
      }
      LOG(INFO) << "Test Passed!";
    }
  }
}
