#ifndef CAVS_UTIL_STREAM_EVENT_HANDLE_POOL_H_
#define CAVS_UTIL_STREAM_EVENT_HANDLE_POOL_H_

#include "cavs/util/macros_gpu.h"

#include <vector>

class StreamEventHandlePool {
 public:
  ~StreamEventHandlePool() {
    for (auto& s : stream_pool_)
      checkCudaError(cudaStreamDestroy(s));
    for (auto& e : event_pool_)
      checkCudaError(cudaEventDestroy(e));
    for (auto iter : handle_pool_)
      checkCublasError(cublasDestroy(iter.second));
  }
  static int GenNewStreamID() {
    cudaStream_t s; 
    checkCudaError(cudaStreamCreate(&s));
    Get()->stream_pool_.push_back(s);
    return Get()->stream_pool_.size()-1;
  }
  static int GenNewEventID() {
    cudaEvent_t e;
    checkCudaError(cudaEventCreate(&e));
    Get()->event_pool_.push_back(e);
    return Get()->event_pool_.size()-1;
  }
  static cublasHandle_t GetCublasHandle(int stream_id) {
    CHECK(stream_id < Get()->stream_pool_.size());
    if (Get()->handle_pool_.find(stream_id) == Get()->handle_pool_.end()) {
      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle));
      checkCublasError(cublasSetStream(handle, GetCudaStream(stream_id)));
      Get()->handle_pool_.emplace(stream_id, handle);
      return handle;
    }else {
      return Get()->handle_pool_.at(stream_id);
    }
  }
  static cudaStream_t GetCudaStream(int sid) {
    CHECK(sid < Get()->stream_pool_.size());
    return Get()->stream_pool_[sid];
    //if (sid < Get()->stream_pool_.size())
      //return Get()->stream_pool_[sid];
    //else
      //return NULL;
  }
  static cudaEvent_t GetCudaEvent(int eid) {
    if (eid < Get()->event_pool_.size())
      return Get()->event_pool_[eid];
    else
      return NULL;
  }

 private:
  static StreamEventHandlePool* Get() {
    static StreamEventHandlePool p; 
    return &p;
  }
  std::vector<cudaStream_t>  stream_pool_;
  std::vector<cudaEvent_t>  event_pool_;
  std::unordered_map<int, cublasHandle_t>  handle_pool_;
};

#endif

