#include "cavs/backend/op_impl.h"
#include "cavs/util/logging.h"

using std::string;
namespace backend {

namespace op_factory {

typedef std::unordered_map<string, 
                           OpImplRegister::Factory> OpImplRegistry;
static OpImplRegistry* GlobalOpImplRegistry() {
  static OpImplRegistry* global_op_impl_registry = new OpImplRegistry();
  return global_op_impl_registry;
}
void OpImplRegister::InitInternal(const string& name,
                              Factory factory) {
  GlobalOpImplRegistry()->insert(std::make_pair(
      name, factory));
}

} //namespace op_factory

OpImpl* CreateOp(const OpDef& def) {
  const string& key = op_factory::Key(def).ToString();
  if (op_factory::GlobalOpImplRegistry()->count(key) == 0)
    return NULL;
  else
    return op_factory::GlobalOpImplRegistry()->at(key)(def);
}

#define INSTANTIATE_GETSINGLEARG(T, fieldname)      \
  template<>                                        \
  T OpImpl::GetSingleArg<T>(const string& key) {    \
    for (auto& attr : op_def_.attr()){              \
      if (attr.name() == key) {                     \
        /*CHECK(attr.value().has_##fieldname());*/  \
        return attr.value().fieldname();            \
      }                                             \
    }                                               \
    LOG(FATAL) << key << " NOT FOUND";              \
  }                                                 \
  template<>                                              \
  T OpImpl::GetSingleArg<T>(const string& key, T value) { \
    for (auto& attr : op_def_.attr()){                    \
      if (attr.name() == key) {                           \
        return attr.value().fieldname();                  \
      }                                                   \
    }                                                     \
    return value;                                         \
  }

INSTANTIATE_GETSINGLEARG(float, f)
INSTANTIATE_GETSINGLEARG(int, i)

} //namespace backend
