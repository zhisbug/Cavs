#include "cavs/backend/functor_reduction.cuh"
#include "cavs/backend/cublas_wrapper.h"
#include "cavs/util/logging.h"
#include "cavs/util/macros_gpu.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <iostream>

using namespace backend;

int main() {
  /*
  thrust::host_vector<float> h_vec(10);
  thrust::generate(h_vec.begin(), h_vec.end(), rand);
  thrust::device_vector<float> d_vec(h_vec);
  thrust::host_vector<int> h_idx_verify(1);
  thrust::device_vector<int> d_idx_verify(1);
  thrust::device_vector<float> d_out(1);
  thrust::device_vector<int> d_idx(1);
  thrust::host_vector<int> h_idx(1);

  for (int i = 0; i < 10; i++)
    LOG(INFO) << "h[" << i << "]:" << h_vec[i];
  BatchedArgmax(thrust::raw_pointer_cast(d_out.data()),
      thrust::raw_pointer_cast(d_idx.data()),
      thrust::raw_pointer_cast(d_vec.data()),
      10, 1);
  ArgmaxCublasWrapper<float>(10, thrust::raw_pointer_cast(d_vec.data()),
      thrust::raw_pointer_cast(d_idx_verify.data()));

  thrust::copy(d_idx.begin(), d_idx.end(), h_idx.begin());
  thrust::copy(d_idx_verify.begin(), d_idx_verify.end(), h_idx_verify.begin());
  LOG(INFO) << "h[" << 0 << "]:" << h_idx[0];
  LOG(INFO) << "h_verify[" << 0 << "]:" << h_idx_verify[0]-1;
  */

  for (int Batch = 1; Batch <= 1 << 10; Batch <<= 1) {
    for (int N = 3; N <= 1 << 11; N <<= 1) {
      thrust::host_vector<float> h_vec(Batch*N);
      thrust::generate(h_vec.begin(), h_vec.end(), rand);
      thrust::device_vector<float> d_vec(h_vec);
      thrust::host_vector<int>     h_idx_verify(Batch);
      thrust::device_vector<int>   d_idx_verify(Batch);
      thrust::device_vector<float> d_out(Batch);
      thrust::device_vector<int>   d_idx(Batch);
      thrust::host_vector<int>     h_idx(Batch);
      LOG(INFO) << "Testing with N = " << N
                << "\tand Batch = " << Batch << "\t...";
      BatchedArgmax(thrust::raw_pointer_cast(d_out.data()),
          thrust::raw_pointer_cast(d_idx.data()),
          thrust::raw_pointer_cast(d_vec.data()),
          N, Batch);

      thrust::copy(d_idx.begin(), d_idx.end(), h_idx.begin());
      for (int i = 0; i < Batch; i++) {
        ArgmaxCublasWrapper<float>(N, (float*)(thrust::raw_pointer_cast(d_vec.data()))+i*N,
            (int*)(thrust::raw_pointer_cast(d_idx_verify.data()))+i);
      }
      thrust::copy(d_idx_verify.begin(), d_idx_verify.end(), h_idx_verify.begin());
      for (int i = 0; i < Batch; i++) {
        CHECK((h_idx[i] == h_idx_verify[i]-1))
             << "h_idx[" << i << "]: "
             << h_idx[i]
             << "\th_idx_verify[" << i << "]: "
             << h_idx_verify[i]-1;
        
      }
      LOG(INFO) << "Test Passed!";
    }
  }
}
