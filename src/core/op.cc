/*
#include "op.h"

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
*/
