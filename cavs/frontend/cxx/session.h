#ifndef CAVS_FRONTEND_CXX_SESSION_H_
#define CAVS_FRONTEND_CXX_SESSION_H_

#include "cavs/frontend/c_api.h"

#include <string>

using std::string;

class Session {
 public:
  Session(string name = "SimpleSession") {
    s = F_NewSession(name);
    cavs::OpChainDef op_chain_def;
    Chain::Default()->Finalize(&op_chain_def);
    string serial_def;
    op_chain_def.SerializeToString(&serial_def_);
    F_SetOpChainOp(s, serial_def.c_str(), serial_def.length());
  }

  void Run() {
    F_Run(s);
  }

 private:
  F_Session* s;
};

#endif
