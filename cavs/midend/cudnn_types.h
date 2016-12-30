#ifndef CAVS_MIDEND_CUDNN_TYPES_H_
#define CAVS_MIDEND_CUDNN_TYPES_H_

#include "cavs/util/macros_gpu.h"

namespace midend {

template <class T>
struct DataTypeToCudnnType {

};

#define MATCH_TYPE_TO_CUDNN_TYPE(TYPE, ENUM)  \
  template <>                           \
  struct DataTypeToCudnnType<TYPE> {    \
    static const cudnnDataType_t value = ENUM; \
  }

MATCH_TYPE_TO_CUDNN_TYPE(float, CUDNN_DATA_FLOAT);
MATCH_TYPE_TO_CUDNN_TYPE(double, CUDNN_DATA_DOUBLE);

#undef MATCH_TYPE_TO_CUDNN_TYPE

} //namespace midend 

#endif
