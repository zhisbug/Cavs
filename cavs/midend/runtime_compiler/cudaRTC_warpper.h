#include "cavs/util/macros_gpu.h"

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

#define checkCUDADriverError(stmt)            \
    do {                                      \
      CUresult err = (stmt);                  \
      if (err != CUDA_SUCCESS) {              \
        const char* pStr;                     \
        cuGetErrorString(err, &pStr);         \
        LOG(FATAL) << "CUDA Driver failure: " \
                   << *pStr;                  \
      }                                       \
    }while(0)


namespace midend {
namespace cudaRTC {

class CudaRTCFunction {
 public:
  CudaRTCFunction() : module_loaded_(false), kernel_(NULL) {
    flags_num_ = 1;
    compiler_flags_ = {"--gpu-architecture=compute_52"};
  }
  ~CudaRTCFunction() {
    if (module_loaded_) {
      checkCUDADriverError(cuModuleUnload(module_));
    }
  }
  void Compile(const std::string& name, const std::string& src) {
    nvrtcProgram prog;
    checkNVRTCError(nvrtcCreateProgram(&prog, src.c_str(),
                                       ("cavs_" + name + ".cu").c_str(),
                                       0, NULL, NULL));
    nvrtcResult compile_result = nvrtcCompileProgram(prog, flags_num_, compiler_flags_);
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
    for (auto& o : outputs) args.push_back(o);
    for (auto& i : inputs)  args.push_back(i);
    args.push_back((void*)num_elements);
    checkCUDADriverError(cuLaunchKernel(kernel_,
          gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
          0, NULL, args.data(), 0));
  }

 private:
  CUmodule module_;
  bool module_loaded_;
  CUfunction kernel_;
  const char *compiler_flags_[];
  int flags_num_;
};

} //namespace cudaRTC
} //namespace midend
