#include "cavs/kernels/elementwise_ops_common.h"
#include "cavs/kernels/elementwise_ops.h"
#include "cavs/core/logging.h"

using namespace cavs;

class AddOpTest : public OpTestBase {};

int main() {
    OpDef op_def;
    OpDefBuilder("Add").Input("A").Input("B").Output("C").Device("GPU")
        .Finalize(&op_def);
    AddOpTest test;
    test.SetOpDef(node_def);
    test.AddInputFromVector<float>(TensorShape({1, 2, 2, 3}),
                            {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
    test.AddInputFromVector<float>(TensorShape({}), {0.2});

    test.InitOp();
    test.RunTest();
    vector output;
    test.FetchOutput("C", output);
    LOG(INFO) << output;
    return 0;
}
