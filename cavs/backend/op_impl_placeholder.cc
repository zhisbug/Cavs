#include "cavs/backend/op_impl_placeholder.h"

namespace backend {

REGISTER_OP_IMPL_BUILDER(Key("Placeholder").Device("CPU"), PlaceholderOpImpl);

} //namespace backend


