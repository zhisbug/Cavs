#include "cavs/backend/functor_sort_scan.cuh"
#include "cavs/util/logging.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <iostream>

using namespace backend;
/*using namespace std;*/


int main() {
  //10k documents
  thrust::host_vector<int> h_vec(1<<3);
  thrust::generate(h_vec.begin(), h_vec.end(), rand);
  for (int i = 0; i < h_vec.size(); i++)
    LOG(INFO) << "h_vec[" << i << "]: " << h_vec[i];
  thrust::device_vector<int> d_vec = h_vec;
  /*thrust::sort(d_vec.begin(), d_vec.end());*/
  int threadsPerBlock = 1<<3;
  int blocksPerGrid = 1;
  bool direction = true;//small first
  BatchedMergeSort< int, 1<<3 ><<<blocksPerGrid, threadsPerBlock>>>(
      thrust::raw_pointer_cast(d_vec.data()),
      1, 1<<3, direction);
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  LOG(INFO);
  for (int i = 0; i < h_vec.size(); i++)
    LOG(INFO) << "h_vec[" << i << "]: " << h_vec[i];
}
