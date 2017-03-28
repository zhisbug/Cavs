#include "cavs/backend/op_impl_placeholder.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

struct CUDAMemCopy {
  static void Compute(void* out, void* in, size_t n) {
    checkCudaError(cudaMemcpy(out, in, n, cudaMemcpyHostToDevice));
  }
};

struct BinaryReader {
  static void Compute(void* buf, const char* filename, size_t n) {
    CHECK(buf);
    FILE *fp = fopen(filename,"rb");
    if (!fp)
      LOG(FATAL) << "file[" << filename << "] does not exists";
    CHECK(fread(buf, sizeof(char), n, fp) == n);
    fclose(fp);
  }
};

REGISTER_OP_IMPL_BUILDER(Key("Placeholder").Device("GPU"), PlaceholderOpImpl<CUDAMemCopy>);
REGISTER_OP_IMPL_BUILDER(Key("Data").Label("BinaryReader").Device("GPU"), DataOpImpl<BinaryReader, CUDAMemCopy, float>);

} //namespace backend

