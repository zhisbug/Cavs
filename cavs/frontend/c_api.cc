#include "c_api.h"

#include "cavs/midend/session.h"
#include "cavs/midend/op_chain_def.pb.h"

using cavs::SessionBase;
using cavs::OpChainDef;
using cavs::GetSession;

struct F_Session {
  SessionBase* session;
};

F_Session* F_NewSession(const char* name, size_t len) {
  string name_str(name, len);
  SessionBase* sess = GetSession(name_str);
  return new F_Session{sess};
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
