#ifndef CAVS_UTIL_MPI_TYPES_H_
#define CAVS_UTIL_MPI_TYPES_H_

#include <mpi.h>

template <class T>
struct DataTypeToMPIType {

};

#define MATCH_TYPE_TO_MPI_TYPE(TYPE, ENUM)  \
  template <>                               \
  struct DataTypeToMPIType<TYPE> {          \
    static const MPI_Datatype value = ENUM; \
  }

MATCH_TYPE_TO_MPI_TYPE(float,  MPI_FLOAT);
MATCH_TYPE_TO_MPI_TYPE(double, MPI_DOUBLE);

#undef MATCH_TYPE_TO_MPI_TYPE

#endif
