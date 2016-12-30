#ifndef CAVS_MIDEND_TYPES_H_
#define CAVS_MIDEND_TYPES_H_

#include "cavs/midend/types.pb.h"

namespace midend {

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

#undef MATCH_TYPE_TO_TYPE

} //namespace midend

#endif
