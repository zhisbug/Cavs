#ifndef CAVS_MIDEND_ALLOCATOR_H_
#define CAVS_MIDEND_ALLOCATOR_H_

#include "cavs/midend/device.pb.h"
#include "cavs/midend/op_def.pb.h"

#include <string>
#include <unordered_map>

using std::string;
using std::unordered_map;

namespace cavs {

class Allocator {
 public:
  virtual string Name() = 0;
  virtual void* AllocateRaw(size_t nbytes) = 0;
  virtual void DeallocateRaw(void* buf) = 0;

  template <typename T>
  T* Allocate(size_t n_elements) {
    void *p = AllocateRaw(n_elements*sizeof(T)); 
    return reinterpret_cast<T*>(p);
  }
  template <typename T>
  void Deallocate(T* buf) {
    if (buf) {
      DeallocateRaw(buf);
    }
  }
};

class TrackingAllocator : public Allocator {
 public:
  explicit TrackingAllocator(Allocator* allocator);
  void* AllocateRaw(size_t nbytes) override;
  void DeallocateRaw(void* buf) override;
  string Name() override { return allocator_->Name(); }
  size_t Capacity() { return capacity_; }
 private:
  Allocator* allocator_;
  size_t capacity_;
  unordered_map<void*, size_t> trace_;
};

Allocator* GetAllocator(const OpDef& def);
Allocator* GetAllocator(const string& device);

#define REGISTER_STATIC_ALLOCATOR(key, alloc)                  \
    REGISTER_STATIC_ALLOCATOR_UNIQ(__COUNTER__, key, alloc)
#define REGISTER_STATIC_ALLOCATOR_UNIQ(ctr, key, alloc)        \
    REGISTER_STATIC_ALLOCATOR_CONCAT(ctr, key, alloc)
#define REGISTER_STATIC_ALLOCATOR_CONCAT(ctr, key, alloc)      \
    static allocator_factory::AllocatorRegister                \
        register_body_##ctr##_allocator(key, alloc)                       

namespace allocator_factory {

class AllocatorRegister {
 public:
  AllocatorRegister(const string& name, Allocator* alloc) {
    InitInternal(name, alloc); 
  }
 private:
  void InitInternal(const string& name, Allocator* alloc); 
};

} //namespace allocator_factory

} //namepsace cavs

#endif
