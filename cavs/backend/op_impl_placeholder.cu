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

struct MPIBinaryReader {
  static void Compute(void* buf, const char* filename, size_t n) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    CHECK(buf);
    FILE *fp = fopen(filename,"rb");
    if (!fp)
      LOG(FATAL) << "file[" << filename << "] does not exists";
    CHECK(fseek(fp, rank*n, SEEK_SET) == 0);
    CHECK(fread(buf, sizeof(char), n, fp) == n);
    fclose(fp);
  }
};

REGISTER_OP_IMPL_BUILDER(Key("Placeholder").Device("GPU"), PlaceholderOpImpl);
REGISTER_OP_IMPL_BUILDER(Key("Data").Label("BinaryReader").Device("GPU"), DataOpImpl<BinaryReader, CUDAMemCopy, float, false>);
REGISTER_OP_IMPL_BUILDER(Key("DataMPI").Label("BinaryReader").Device("GPU"), DataOpImpl<MPIBinaryReader, CUDAMemCopy, float, true>);

} //namespace backend

