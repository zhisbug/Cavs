#include "op_def_builder.h"

#include <glog/logging.h>
#include <gflags/gflags.h>

using namespace cavs;

int main() {
    GraphDef graph_def;
    OpDefBuilder("Add").Input("A").Input("B").Output("C").Device("GPU")
        .AddToGraphDef(&graph_def);
    OpDefBuilder("Add").Input("A").Input("B").Output("C").Device("GPU")
        .AddToGraphDef(&graph_def);
    LOG(INFO) << "\n" << graph_def.DebugString();
    return 0;
}
