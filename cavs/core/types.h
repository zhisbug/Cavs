#ifndef CAVS_CORE_TYPES_H_
#define CAVS_CORE_TYPES_H_

#include "cavs/core/types.pb.h"

namespace cavs {

template <class T>
struct DataTypeToEnum {

};

#define MATCH_TYPE_TO_TYPE(TYPE, ENUM)  \
  template <>                           \
  struct DataTypeToEnum<TYPE> {         \
    static const DataType value = ENUM; \
  }

MATCH_TYPE_TO_TYPE(float, DT_FLOAT);
MATCH_TYPE_TO_TYPE(double, DT_DOUBLE);
MATCH_TYPE_TO_TYPE(int, DT_INT32);

} //namespace cavs

#endif
