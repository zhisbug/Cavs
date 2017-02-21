#include "cavs/util/op_util.h"
#include "cavs/util/logging.h"

using std::string;
using std::vector;

#define INSTANTIATE_GETSINGLEARG(T, fieldname)      \
  template<>                                        \
  T GetSingleArg<T>(const OpDef& op_def,            \
      const string& key) {                          \
    for (auto& attr : op_def.attr()){               \
      if (attr.name() == key) {                     \
        /*CHECK(attr.value().has_##fieldname());*/  \
        return attr.value().fieldname();            \
      }                                             \
    }                                               \
    LOG(FATAL) << key << " NOT FOUND\n"             \
               << op_def.DebugString();             \
  }                                                 \
  template<>                                        \
  T GetSingleArg<T>(const OpDef& op_def,            \
      const string& key, T value) {                 \
    for (auto& attr : op_def.attr()){               \
      if (attr.name() == key) {                     \
        return attr.value().fieldname();            \
      }                                             \
    }                                               \
    return value;                                   \
  }                                                 \
  template<>                                             \
  vector<T> GetListArg<T>(const OpDef& op_def,           \
      const string& key) {                               \
    vector<T> ret;                                       \
    for (auto& attr : op_def.attr()) {                   \
      if (attr.name() == key) {                          \
        for (auto& m : attr.value().list().fieldname())  \
          ret.push_back(m);                              \
      }                                                  \
    }                                                    \
    return ret;                                          \
  }                                                       

INSTANTIATE_GETSINGLEARG(float, f)
INSTANTIATE_GETSINGLEARG(int, i)
