#ifndef CAVS_UTIL_TIMING_H_
#define CAVS_UTIL_TIMING_H_

#include "cavs/util/macros_gpu.h"

#include <unordered_map>
#include <string>
#include <utility>

class Timing {
 public:
  static void TimingBegin(const std::string& name) {
    if (Get()->status_.find(name) == Get()->status_.end())
      Get()->status_[name] = true;
    else {
      CHECK(!Get()->status_[name]);
      Get()->status_[name] = true;
    }

    if (Get()->event_.find(name) == Get()->event_.end()) {
      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start));
      checkCudaError(cudaEventCreate(&stop));
      Get()->event_[name] = std::make_pair(start, stop);
    }

    if (Get()->time_in_ms_.find(name) == Get()->time_in_ms_.end())
      Get()->time_in_ms_[name] = 0;
    cudaEvent_t start = Get()->event_[name].first;
    checkCudaError(cudaEventRecord(start));
  }
  static void TimingEnd(const std::string& name) {
    CHECK(Get()->status_.find(name) != Get()->status_.end());
    CHECK(Get()->status_[name]);
    Get()->status_[name] = false;
    CHECK(Get()->event_.find(name) != Get()->event_.end());
    cudaEvent_t start = Get()->event_[name].first;
    cudaEvent_t stop = Get()->event_[name].second;
    CHECK(start);
    CHECK(stop);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    float ms = 0;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop));
    CHECK(Get()->time_in_ms_.find(name) != Get()->time_in_ms_.end());
    Get()->time_in_ms_[name] += ms;
  }

  static float TimeInMs(const std::string& name) {
    CHECK(Get()->time_in_ms_.find(name) != Get()->time_in_ms_.end()); 
    return Get()->time_in_ms_[name];
  }
  static void Reset(const std::string& name) {
    CHECK(Get()->time_in_ms_.find(name) != Get()->time_in_ms_.end()); 
    Get()->time_in_ms_[name] = 0;
  }

 private:
  static Timing* Get() {
    static Timing t; 
    return &t;
  }
  std::unordered_map<std::string, bool> status_;//0 null; 1:timing
  std::unordered_map<std::string, std::pair<cudaEvent_t, cudaEvent_t>> event_;
  std::unordered_map<std::string, float> time_in_ms_;

};

#endif
