#ifndef CAVS_BACKEND_CUDARTC_WRAPPER_H_
#define CAVS_BACKEND_CUDARTC_WRAPPER_H_

#include "cavs/util/macros_gpu.h"

#include <cuda.h>
#include <nvrtc.h>
#include <vector>

#define checkNVRTCError(stmt)                   \
    do {                                        \
      nvrtcResult err = (stmt);                 \
      if (err != NVRTC_SUCCESS) {               \
        LOG(FATAL) << "NVRTC failure: "         \
                   << nvrtcGetErrorString(err); \
      }                                         \
    }while(0)

#define checkCUDADriverError(stmt)              \
    do {                                        \
      CUresult err = (stmt);                    \
      if (err != CUDA_SUCCESS) {                \
        const char* pStr;                       \
        cuGetErrorString(err, &pStr);           \
        LOG(FATAL) << "CUDA Driver failure: "   \
                   << *pStr;                    \
      }                                         \
    }while(0)


namespace backend {
namespace RTC {

class CudaRTCWrapper {
 public:
  CudaRTCWrapper() : module_loaded_(false), kernel_(NULL) {}
  ~CudaRTCWrapper() {
    if (module_loaded_) {
      checkCUDADriverError(cuModuleUnload(module_));
    }
  }
  void Compile(const std::string& name, const std::string& src) {
    nvrtcProgram prog;
    checkNVRTCError(nvrtcCreateProgram(&prog, src.c_str(),
                                       ("cavs_" + name + ".cu").c_str(),
                                       0, NULL, NULL));
    const int flags_num = 2;
    const char *compiler_flags[] =
      {{"--gpu-architecture=compute_52"}, {"--fmad=false"}};
    nvrtcResult compile_result = nvrtcCompileProgram(prog, flags_num, compiler_flags);
    if (compile_result != NVRTC_SUCCESS) {
      size_t log_size;
      checkNVRTCError(nvrtcGetProgramLogSize(prog, &log_size));
      std::vector<char> nvrtc_log(log_size);
      checkNVRTCError(nvrtcGetProgramLog(prog, nvrtc_log.data()));
      LOG(FATAL) << "Compile Error:\n"
                 << nvrtcGetErrorString(compile_result)
                 << "\nKernel Source:\n"
                 << nvrtc_log.data();
    }

    size_t ptx_size;
    checkNVRTCError(nvrtcGetPTXSize(prog, &ptx_size));
    std::vector<char> nvrtc_ptx(ptx_size);
    checkNVRTCError(nvrtcGetPTX(prog, nvrtc_ptx.data()));
    checkNVRTCError(nvrtcDestroyProgram(&prog));

    if (module_loaded_) {
      checkCUDADriverError(cuModuleUnload(module_));
    }
    checkCUDADriverError(cuModuleLoadDataEx(&module_, nvrtc_ptx.data(), 0, 0, 0));
    module_loaded_ = true;
    checkCUDADriverError(cuModuleGetFunction(&kernel_, module_, name.c_str()));
  }

  void Launch(const std::vector<void*>& outputs,
              const std::vector<void*>& inputs,
              const std::vector<int>& outputs_size,
              const std::vector<int>& inputs_size,
              unsigned int num_elements,
              unsigned int gridDimX,
              unsigned int gridDimY,
              unsigned int gridDimZ,
              unsigned int blockDimX,
              unsigned int blockDimY,
              unsigned int blockDimZ) {
    CHECK(module_loaded_);
    CHECK(kernel_);
    std::vector<void*> args;
    for (int i = 0; i < outputs.size(); i++) args.push_back((void*)(&outputs[i]));
    for (int i = 0; i < inputs.size(); i++)  args.push_back((void*)(&inputs[i]));
    for (int i = 0; i < outputs_size.size(); i++) args.push_back((void*)&outputs_size[i]);
    for (int i = 0; i < inputs_size.size(); i++) args.push_back((void*)&inputs_size[i]);
    args.push_back(&num_elements);
    checkCUDADriverError(cuLaunchKernel(kernel_,
          gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
          0, NULL, args.data(), 0));
  }

 private:
  CUmodule module_;
  bool module_loaded_;
  CUfunction kernel_;
};

} //namespace RTC
} //namespace backend

#endif
