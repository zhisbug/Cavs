#include "cavs/midend/op.h"
#include "cavs/util/logging.h"

namespace midend {

OpContext::OpContext(const OpDef& op_def, SessionBase* sb) {
  for (const string& input : op_def.input()) {
    const Tensor* t = sb->GetTensor(input); 
    inputs_.push_back(t);
  }
  for (const string& output : op_def.output()) {
    const Tensor* t = sb->GetTensor(output);
    if (!t)
      for (const OpDef::AttrDef& attr : op_def.attr()) {
        if (attr.name() == "shape") {
          CHECK(attr.value().has_list());
          TensorShape shape(attr.value().list()); 
          Allocator* alloc = GetAllocator(op_def); 
          CHECK_NOTNULL(alloc);
          Tensor out(output, alloc, op_def.dtype(), std::move(shape));
          sb->InsertTensor(out);
          break;
        }
      }
    t = sb->GetTensor(output);
    CHECK_NOTNULL(t);
    outputs_.push_back(const_cast<Tensor*>(t));
  }
}

namespace op_factory {

typedef std::unordered_map<string, 
                           OpRegister::Factory> OpRegistry;
static OpRegistry* GlobalOpRegistry() {
    static OpRegistry* global_op_registry = new OpRegistry();
    return global_op_registry;
}
void OpRegister::InitInternal(const string& name,
                                    Factory factory) {
    GlobalOpRegistry()->insert(std::make_pair(
        name, factory));
}

} //namespace op_factory

Op* CreateOp(const OpDef& def) {
    const string key = op_factory::Key(def).ToString();
    if (op_factory::GlobalOpRegistry()->count(key) == 0)
        return NULL;
    else
        return op_factory::GlobalOpRegistry()->at(key)(def);
}


} //namespace midend
