#include "cavs/backend/op_def_builder.h"
#include "cavs/midend/op_test.h"
#include "cavs/util/logging.h"

using namespace midend;
using namespace backend;
using namespace midend::test;

class AddOptest : public OpTest {
 public:
  AddOptest(const OpDef& def) : OpTest(def) {}
};

int main() {
  OpDef op_def;
  auto shape = {2, 3};
  OpDefBuilder("Add").Input("A").Input("B").Output("C")
    .Device("GPU").Shape(shape).Finalize(&op_def);
  AddOptest add_test(op_def);
  add_test.AddTensorFromVector<float>("A", 
      TensorShape(shape), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  add_test.AddTensorFromVector<float>("B", 
      TensorShape(shape), {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  add_test.RunTest();
  vector<float> output;
  add_test.FetchTensor("C", &output);

  for (auto& i : output)
    LOG(INFO) << i;
  return 0;
}
