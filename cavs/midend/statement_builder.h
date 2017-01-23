#ifndef CAVS_MIDEND_STATEMENT_BUILDER_H_
#define CAVS_MIDEND_STATEMENT_BUILDER_H_

#include "cavs/midend/statement.h"
#include "cavs/midend/dep_graph.h"
#include "cavs/proto/op_def.pb.h"

#include <vector>

namespace midend {

Statement BuildStatement(const Node* node);
Statement BuildStatement(const OpDef& op_def);
BasicBlock BuildBasicBlock(const std::vector<Statement>& stmts);

} //namespace midend

#endif


