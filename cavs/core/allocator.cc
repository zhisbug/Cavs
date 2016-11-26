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
REGISTER_STATIC_ALLOCATOR("CPU", cpu_allocator());

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

typedef std::unordered_map<string, Allocator*> AllocatorRegistry;
static AllocatorRegistry* GlobalAllocatorRegistry() {
    static AllocatorRegistry* global_allocator_registry = new AllocatorRegistry();
    return global_allocator_registry;
}

Allocator* GetAllocator(const OpDef& def) {
    DeviceType dev = def.device();
    string dev_name;
    if (dev == GPU)
        dev_name = "GPU";
    else
        dev_name = "CPU";
    if (GlobalAllocatorRegistry()->count(dev_name) == 0)
        return NULL;
    else
        return GlobalAllocatorRegistry()->at(dev_name);
}

namespace allocator_factory {

void AllocatorRegister::InitInternal(const string& name, Allocator* alloc) {
    GlobalAllocatorRegistry()->insert(std::make_pair(name, alloc));
}

} //namespace allocator_factory

} //namespace cavs
