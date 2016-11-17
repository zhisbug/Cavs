#include "allocator.h"

#include "macros_gpu.h"

namespace cavs {

class GPUAllocator : public Allocator {
 public:
  GPUAllocator() : Allocator() {}    
  string Name() override { return "CPU"; }

  void* AllocateRaw(size_t nbytes) {
    void* ptr = NULL;
    checkCudaErrors(cudaMalloc(&ptr, nbytes)); 
    return ptr;
  }
  bool DeallocateRaw(void* buf) {
    checkCudaErrors(cudaFree(buf));
    return true;
  }
};

Allocator* gpu_allocator() {
  static GPUAllocator gpu_alloc;
  return &gpu_alloc;
}

}
