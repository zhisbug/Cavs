#include "cavs/backend/functor_sort_scan.cuh"
#include "cavs/util/logging.h"
#include "cavs/util/macros_gpu.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <iostream>

using namespace backend;

const int SHARE_SIZE_LIMIT = 1 << 13;

int main() {
  //8k documents
  for (int Batch = 1; Batch <= 1 << 10; Batch <<= 1) {
    for (int N = 5; N <= 1 << 10; N <<= 1) {
      thrust::host_vector<float> h_vec(Batch*N, 2.f);
      thrust::device_vector<float> d_vec(h_vec);
      thrust::host_vector<float> h_vec_verify(h_vec);
      thrust::device_vector<float> d_vec_verify(h_vec);
      LOG(INFO) << "Testing with N = " << N
                << "\tand Batch = " << Batch << "\t...";
      int threadsPerBlock = N;
      int blocksPerGrid = Batch;
      CHECK(N <= SHARE_SIZE_LIMIT);
      CHECK(N == threadsPerBlock);//it is assumed in current implementation

      BatchedScan<SHARE_SIZE_LIMIT><<<blocksPerGrid, threadsPerBlock>>>(
          thrust::raw_pointer_cast(d_vec.data()),
          thrust::raw_pointer_cast(d_vec.data()),
          N);

      checkCudaError(cudaDeviceSynchronize());
      checkCudaError(cudaGetLastError());
      thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

      for (int i = 0; i < Batch; i++) {
        thrust::inclusive_scan(
            d_vec_verify.begin()+i*N, d_vec_verify.begin()+(i+1)*N, 
            d_vec_verify.begin()+i*N);
      }
      thrust::copy(d_vec_verify.begin(), d_vec_verify.end(), h_vec_verify.begin());
      for (int i = 0; i < Batch; i++) {
        for (int j = 0; j < N; j++) {
          CHECK((h_vec[i*N+j] == h_vec_verify[i*N+j]))
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
