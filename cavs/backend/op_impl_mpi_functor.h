#ifndef CAVS_BACKEND_OP_IMPL_MPI_FUNCTOR_H_
#define CAVS_BACKEND_OP_IMPL_MPI_FUNCTOR_H_

#include "cavs/backend/op_impl.h"
#include "cavs/util/mpi_types.h"

#define checkMPIError(stmt)                            \
  do {                                                 \
    char err_buffer[MPI_MAX_ERROR_STRING];             \
    int result_len;                                    \
    int ierr = (stmt);                                 \
    if (ierr != MPI_SUCCESS) {                         \
      MPI_Error_string(ierr, err_buffer, &result_len); \
      LOG(INFO) << err_buffer;                         \
      MPI_Finalize();                                  \
    }                                                  \
  }while(0) 

namespace backend {

template <typename T>
struct MPIBcastFunctor {
  inline static void Compute(void* buf, int count, int root) {
    checkMPIError(MPI_Bcast(buf, count,
          DataTypeToMPIType<T>::value,
          root, MPI_COMM_WORLD));
  }
};

template <typename T>
struct MPIAllgatherFunctor {
  inline static void Compute(const void* sendbuf, int sendcount, 
      void* recvbuf, int recvcount) {
    checkMPIError(MPI_Allgather(
          sendbuf, sendcount, DataTypeToMPIType<T>::value,
          recvbuf, recvcount, DataTypeToMPIType<T>::value,
          MPI_COMM_WORLD));
  }
};

//MPI is currently run on CPU and the communication is global
//CUDA-aware MPI will be supported later
//For MPI_Allreduce operator, only MPI_SUM is supported.
template <typename T>
struct MPIAllReduceFunctor {
 public:
  inline static void Compute(const void* sendbuf,
      void* recvbuf, int count) {
    if (reinterpret_cast<int64_t>(sendbuf) == 
        reinterpret_cast<int64_t>(recvbuf)) {
      checkMPIError(MPI_Allreduce(MPI_IN_PLACE, recvbuf,
            count, DataTypeToMPIType<T>::value,
            MPI_SUM, MPI_COMM_WORLD));
    }else {
      checkMPIError(MPI_Allreduce(sendbuf, recvbuf,
            count, DataTypeToMPIType<T>::value,
            MPI_SUM, MPI_COMM_WORLD));
    }
  }
};

} //namespace backend

#endif
