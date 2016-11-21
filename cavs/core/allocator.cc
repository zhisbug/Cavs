#include "cavs/core/allocator.h"
#include "cavs/core/logging.h"

namespace cavs {

class CPUAllocator : public Allocator {
 public:
  CPUAllocator() : Allocator() {}    
  string Name() override { return "CPU"; }

  void* AllocateRaw(size_t nbytes) {
    return malloc(nbytes); 
  }
  void DeallocateRaw(void* buf) {
    free(buf);
  }
};


Allocator* cpu_allocator() {
  static CPUAllocator cpu_alloc;
  return &cpu_alloc;
}


TrackingAllocator::TrackingAllocator(Allocator* allocator)
    : allocator_(allocator), capacity_(0) {}

void* TrackingAllocator::AllocateRaw(size_t nbytes) {
  void* ptr = allocator_->AllocateRaw(nbytes);
  capacity_ += nbytes;
  trace_[ptr] = nbytes;
  return ptr;
}

void TrackingAllocator::DeallocateRaw(void *buf) {
  CHECK(trace_.find(buf) != trace_.end());
  capacity_ -= trace_[buf];
  trace_.erase(buf);
}

}
