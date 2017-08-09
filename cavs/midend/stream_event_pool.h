#ifndef CAVS_MIDEND_STREAM_EVENT_POLL_H_
#define CAVS_MIDEND_STREAM_EVENT_POLL_H_

#include "cavs/util/macros_gpu.h"

#include <vector>

namespace midend {

class StreamEventPool {
 public:
  ~StreamEventPool() {
    for (auto& s : stream_pool_)
      checkCudaError(cudaStreamDestroy(s));
    for (auto& e : event_pool_)
      checkCudaError(cudaEventDestroy(e));
  }
  static int GetNewStream() {
    cudaStream_t s; 
    checkCudaError(cudaStreamCreate(&s));
    Get()->stream_pool_.push_back(s);
    return Get()->stream_pool_.size()-1;
  }
  static int GetNewEvent() {
    cudaEvent_t e;
    checkCudaError(cudaEventCreate(&e));
    Get()->event_pool_.push_back(e);
    return Get()->event_pool_.size()-1;
  }
  static cudaStream_t GetStream(int sid) {
    if (sid < Get()->stream_pool_.size())
      return Get()->stream_pool_[sid];
    else
      return NULL;
  }
  static cudaEvent_t GetEvent(int eid) {
    if (eid < Get()->event_pool_.size())
      return Get()->event_pool_[eid];
    else
      return NULL;
  }

 private:
  static StreamEventPool* Get() {
    static StreamEventPool p; 
    return &p;
  }
  std::vector<cudaStream_t>  stream_pool_;
  std::vector<cudaEvent_t>  event_pool_;
};

} //namespace midend

#endif

