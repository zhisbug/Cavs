#include "cavs/midend/allocator.h"
#include "cavs/midend/macros_gpu.h"
#include "cavs/midend/devices.h"

namespace cavs {

class GPUAllocator : public Allocator {
 public:
  GPUAllocator() 
      : Allocator(DeviceTypeToString(GPU), GPU) {}    
  void* AllocateRaw(size_t nbytes) override {
    void* ptr = NULL;
    checkCudaError(cudaMalloc(&ptr, nbytes)); 
    return ptr;
  }
  void DeallocateRaw(void* buf) override {
    checkCudaError(cudaFree(buf));
  }
};

Allocator* gpu_allocator() {
  static GPUAllocator gpu_alloc;
  return &gpu_alloc;
}
REGISTER_STATIC_ALLOCATOR("GPU", gpu_allocator());

}
