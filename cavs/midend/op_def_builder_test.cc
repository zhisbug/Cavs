#include "cavs/midend/op_def_builder.h"
#include "cavs/util/logging.h"

using namespace cavs;

int main() {
  OpChainDef op_chain_def;
  OpDefBuilder("Add").Input("A").Input("B").Output("C").Device("GPU")
      .AddToOpChainDef(&op_chain_def);
  LOG(INFO) << "\n" << op_chain_def.DebugString();
  return 0;
}
