#include "c_api.h"

#include "cavs/midend/session.h"
#include "cavs/midend/op_chain_def.pb.h"

using cavs::SessionBase;

struct F_Session {
  SessionBase* session;
};

F_Session* F_NewSession(const string& name) {
  SessionBase* sess = GetSession(name);
  return new F_Session(sess);
}

void F_SetOpChainOp(F_Session* s, 
                    const void* proto, size_t len) {
  OpChainDef def;
  def.ParseFromArray(proto, len);
  s->session->SetOpChainDef(def);
}

void F_Run(F_Session* s) {
  s->session->Run();
}
