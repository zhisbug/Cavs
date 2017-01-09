#include "cavs/backend/op_impl.h"
#include "cavs/util/logging.h"

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

} //namespace backend
