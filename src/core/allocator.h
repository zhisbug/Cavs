#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_

#include <string>

namespace cavs {

class Allocator {
 public:
  virtual ~Allocator() {}
  virtual std::string Name() = 0;
  virtual void* AllocateRaw(size_t nbytes) = 0;
  virtual void* DeallocateRaw(void *buf) = 0;

  template <typename T>
  T* Allocate(size_t n_elements) {
    void *p = AllocateRaw(n_elements*sizeof(T)); 
    return reinterpret_cast<T*>(p);
  }
  template <typename T>
  void* Deallocate(T* buf) {
    if (buf) {
        DeallocateRaw(buf);
    }
  }
};

Allocator* cpu_allocator();
Allocator* gpu_allocator();
Allocator* gpu_pool_allocator();

} //namepsace cavs

#endif
