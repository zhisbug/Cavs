#include "cavs/midend/op_def_builder.h"

namespace cavs {

OpDefBuilder::OpDefBuilder(string op_name) {
    op_def_.set_name(op_name);
}

OpDefBuilder& OpDefBuilder::Input(string input) {
    op_def_.add_input(input);
    return *this;
}

OpDefBuilder& OpDefBuilder::Output(string output) {
    op_def_.add_output(output);
    return *this;
}

OpDefBuilder& OpDefBuilder::Device(string dev) {
    if (dev == "GPU")
        op_def_.set_device(GPU);
    else 
        op_def_.set_device(CPU);
    return *this;
}

void OpDefBuilder::AddToOpChainDef(OpChainDef* op_chain_def) {
    OpDef* op_def = op_chain_def->add_op();
    *op_def = op_def_;
}

void OpDefBuilder::Finalize(OpDef* op_def) {
    *op_def = op_def_;
}

} //namespace cavs
