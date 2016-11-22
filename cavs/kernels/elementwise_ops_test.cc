#include "cavs/core/logging.h"
#include "cavs/core/op_def_builder.h"
#include "cavs/core/op_test.h"

using namespace cavs;
using namespace cavs::test;

class AddOptest : public OpTestBase {
 public:
  AddOptest(const OpDef& def) : OpTestBase(def) {}
};

int main() {
    OpDef op_def;
    OpDefBuilder("Add").Input("A").Input("B").Output("C").Device("GPU")
        .Finalize(&op_def);
    AddOptest add_test(op_def);
    add_test.AddTensorFromVector<float>("A", 
                        TensorShape({2, 3}),
                        {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    add_test.AddTensorFromVector<float>("B", 
                        TensorShape({2, 3}), 
                        {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    add_test.RunTest();
    vector<float> output;
    add_test.FetchTensor("C", output);
    for (auto && i : output)
        LOG(INFO) << i;
    return 0;
}
