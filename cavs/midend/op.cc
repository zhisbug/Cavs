#include "cavs/midend/op.h"
#include "cavs/util/logging.h"

namespace cavs {

OpContext::OpContext(const OpDef& op_def, SessionBase* sb) {
  for (const string& input : op_def.input()) {
    const Tensor* t = sb->GetTensor(input); 
    CHECK_NOTNULL(t);
    inputs_.push_back(t);
  }
  for (const string& output : op_def.output()) {
    Tensor* t = const_cast<Tensor*>(sb->GetTensor(output));
    if (t)
      outputs_.push_back(t);
    else{
      for (const OpDef::AttrDef& attr : op_def.attr()) {
        if (attr.name() == "shape") {
          CHECK(attr.value().has_list());
          TensorShape shape(attr.value().list()); 
          Allocator* alloc = GetAllocator(op_def); 
          CHECK_NOTNULL(alloc);
          t = new Tensor(output, alloc, op_def.out_type(), shape);
          outputs_.push_back(t);
          sb->InsertTensor(t);
          break;
        }
      }
      CHECK_NOTNULL(t);
    }
  }
}

typedef std::unordered_map<string, 
    op_factory::OpRegister::Factory> OpRegistry;

static OpRegistry* GlobalOpRegistry() {
    static OpRegistry* global_op_registry = new OpRegistry();
    return global_op_registry;
}

Op* CreateOp(const OpDef& def) {
    const string key = op_factory::Key(def).ToString();
    if (GlobalOpRegistry()->count(key) == 0)
        return NULL;
    else
        return (GlobalOpRegistry()->at(key))(def);
}

namespace op_factory {

void OpRegister::InitInternal(const string& name,
                                    Factory factory) {
    GlobalOpRegistry()->insert(std::make_pair(
        name, factory));
}

} //namespace op_factory

} //namespace cavs
