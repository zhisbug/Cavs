#include "allocator.h"

namespace cavs {

class CPUAllocator : public Allocator {
 public:
  CPUAllocator() : Allocator() {}    
  string Name() override { return "CPU"; }

  void* AllocateRaw(size_t nbytes) {
    return malloc(nbytes); 
  }
  bool DeallocateRaw(void* buf) {
    free(buf);
    return true;
  }
};


Allocator* cpu_allocator() {
  static CPUAllocator cpu_alloc;
  return &cpu_alloc;
}


TrackingAllocator::TrackingAllocator(Allocator* allocator)
    : allocator_(allocator), capacity_(0) {}

void* TrackingAllocator::AllocateRaw(size_t nbytes) {
  capacity_ += n_elements*sizeof(T);
  trace_[p] = n_elements*sizeof(T);
  return allocator_->AllocateRaw(nbytes);
}

bool TrackingAllocator::DeallocateRaw(void *buf) {
  CHECK(trace_.find((void*)buf) != trace_.end());
  capacity_ -= trace_[(void*)buf];
  trace_.erase((void*)buf);
  return true;
}

}
