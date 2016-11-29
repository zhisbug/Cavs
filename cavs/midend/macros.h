#ifndef CAVS_MIDEND_MACROS_H_
#define CAVS_MIDEND_MACROS_H_

#define DISALLOW_COPY_AND_ASSIGN(TypeName)         \
        TypeName(const TypeName&) = delete;        \
        void operator=(const TypeName&) = delete

#define FORCE_INLINE inline __attribute__((always_inline))

#ifdef __NVCC__
    #define __DEVICE__ __device__
#else
    #define __DEVICE__ 
#endif

#endif
