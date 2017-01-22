#ifndef CAVS_MIDEND_STATEMENT_BUILDER_H_
#define CAVS_MIDEND_STATEMENT_BUILDER_H_

#include "cavs/midend/statement.h"
#include "cavs/midend/dep_graph.h"

namespace midend {

Statement* buildStatement(Node);
BasicBlock* buildBasicBlock(Node);

} //namespace midend

#endif


