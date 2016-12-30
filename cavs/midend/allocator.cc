#include "cavs/midend/allocator.h"
#include "cavs/midend/devices.h"
#include "cavs/util/logging.h"

namespace midend {

class CPUAllocator : public Allocator {
 public:
  CPUAllocator() 
      : Allocator(DeviceTypeToString(CPU), CPU) {}    
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
REGISTER_STATIC_ALLOCATOR(DeviceTypeToString(CPU), cpu_allocator());

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

namespace allocator_factory {

typedef std::unordered_map<string, Allocator*> AllocatorRegistry;
static AllocatorRegistry* GlobalAllocatorRegistry() {
  static AllocatorRegistry* global_allocator_registry = new AllocatorRegistry();
  return global_allocator_registry;
}
void AllocatorRegister::InitInternal(const string& name, Allocator* alloc) {
  GlobalAllocatorRegistry()->insert(std::make_pair(name, alloc));
}

} //namespace allocator_factory

Allocator* GetAllocator(const OpDef& def) {
  DeviceType dev = def.device();
  string dev_name;
  if (dev == GPU)
    dev_name = "GPU";
  else
    dev_name = "CPU";
  return GetAllocator(dev_name);
}

Allocator* GetAllocator(const string& dev) {
  if (allocator_factory::GlobalAllocatorRegistry()->count(dev) == 0)
    return NULL;
  else
    return allocator_factory::GlobalAllocatorRegistry()->at(dev);
}

} //namespace midend 
