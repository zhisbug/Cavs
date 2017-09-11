#include "cavs/midend/allocator.h"
/*#include "cavs/midend/devices.h"*/
#include "cavs/util/macros_gpu.h"
#include "cavs/util/op_util.h"

namespace midend {

class GPUAllocator : public Allocator {
 public:
  GPUAllocator() 
      : Allocator(DeviceTypeToString(GPU), GPU) {}    
  void* AllocateRaw(size_t nbytes) override {
    VLOG(V_DEBUG) << "allocating " << nbytes << " bytes";
    void* ptr = NULL;
    checkCudaError(cudaMalloc(&ptr, nbytes)); 
    checkCudaError(cudaMemset(ptr, 0, nbytes)); 
    CHECK_NOTNULL(ptr);
    return ptr;
  }
  void DeallocateRaw(void* buf) override {
    checkCudaError(cudaFree(buf));
  }
  void InitWithZero(void* buf, size_t nbytes) override {
    checkCudaError(cudaMemsetAsync(buf, 0, nbytes, cudaStreamDefault));
  }
};

Allocator* gpu_allocator() {
  static GPUAllocator gpu_alloc;
  return &gpu_alloc;
}

REGISTER_STATIC_ALLOCATOR("GPU", gpu_allocator());

} //namespace midend
