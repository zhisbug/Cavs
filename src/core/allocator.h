#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_

#include <string>
#include <map>

namespace cavs {

class Allocator {
 public:
  virtual string Name() = 0;
  virtual void* AllocateRaw(size_t nbytes) = 0;
  virtual bool DeallocateRaw(void* buf) = 0;

  template <typename T>
  T* Allocate(size_t n_elements) {
    void *p = AllocateRaw(n_elements*sizeof(T)); 
    return reinterpret_cast<T*>(p);
  }
  template <typename T>
  bool Deallocate(T* buf) {
    if (buf) {
        DeallocateRaw(buf);
    }
    return true;
  }
};

class TrackingAllocator : public Allocator {
 public:
  explicit TrackingAllocator(Allocator* allocator);
  string Name() override { return allocator_->Name(); }
  size_t Capacity() { return capacity_; }
 private:
  Allocator* allocator_;
  size_t capacity_;
  unordered_map<void*, size_t> trace_;
};

Allocator* cpu_allocator();
Allocator* gpu_allocator();
//Allocator* gpu_pool_allocator();

} //namepsace cavs

#endif
