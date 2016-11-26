#include "cavs/core/op.h"
#include "cavs/core/logging.h"

namespace cavs {

Op::Op(const OpDef& def, Session* s) {
    for (const string& input : def.input()) {
        const Tensor* t = s->GetTensor(input); 
        CHECK(t);
        inputs_.push_back(t);
    }

    for (const string& output : def.output()) {
        Tensor* t = const_cast<Tensor*>(s->GetTensor(output));
        if (t)
            outputs_.push_back(t);
        else
            outputs_.push_back(new Tensor());
    }
}

typedef std::unordered_map<string, 
    op_factory::OpRegister::Factory> OpRegistry;

static OpRegistry* GlobalOpRegistry() {
    static OpRegistry* global_op_registry = new OpRegistry();
    return global_op_registry;
}

Op* CreateOp(const OpDef& def, Session *s) {
    const string key = op_factory::Key(def).LowerToString();
    if (GlobalOpRegistry()->count(key) == 0)
        return NULL;
    else
        return (GlobalOpRegistry()->at(key))(def, s);
}

namespace op_factory {

void OpRegister::InitInternal(const string& name,
                                    Factory factory) {
    GlobalOpRegistry()->insert(std::make_pair(
        name, factory));
}

} //namespace op_factory

} //namespace cavs
