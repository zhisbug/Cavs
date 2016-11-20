#include "op.h"

Op::Op(const OpDef* def, Session* s) {
    for (const string& input : def.input()) {
        Tensor* t = s->GetTensor(input); 
        CHECK(t);
        inputs_.push_back(t);
    }

    for (const string& output : def.outptu()) {
        Tensor* t = s->GetTensor(output);
        if (!t)
            t = new Tensor();
        outputs_.push_back(t);
    }
}

typedef std::unordered_multimap<string, 
    op_factory::OpRegister::Factory> OpRegistry;

static OpRegistry* GlobalOpRegistry() {
    static OpRegistry* global_op_registry = new OpRegistry;
    return global_op_registry;
}

namespace op_factory {

void OpRegister::InitInternal(const string& name,
                                    Factory factory) {
    GlobalOpRegistry()->insert(std::make_pair(
        name, factory));
}

}
