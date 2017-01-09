#include "cavs/backend/op_def_builder.h"
#include "cavs/util/logging.h"

using namespace backend;

int main() {
  OpDef op_def;
  OpDefBuilder("Add").Input("A").Input("B").Output("C").Device("GPU")
      .Finalize(&op_def);
  LOG(INFO) << "\n" << op_def.DebugString();
  return 0;
}
