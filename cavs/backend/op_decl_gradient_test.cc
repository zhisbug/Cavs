#include "cavs/backend/op_def_builder.h"
#include "cavs/backend/op_decl.h"
#include "cavs/util/logging.h"

#include <vector>

using namespace std;
using namespace backend;

int main() {
  OpDef op_def;
  OpDefBuilder("Add").Input("A").Input("B").Output("C").Device("GPU")
      .Finalize(&op_def);
  const vector<OpDef>& grads =  MakeGradient(op_def);
  for (auto& grad : grads)
    LOG(INFO) << "\n" << grad.DebugString();
  return 0;
}
