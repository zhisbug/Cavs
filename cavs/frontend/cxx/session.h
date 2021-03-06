#ifndef CAVS_FRONTEND_CXX_SESSION_H_
#define CAVS_FRONTEND_CXX_SESSION_H_

#include "cavs/frontend/c_api.h"
#include "cavs/frontend/cxx/sym.h"
#include "cavs/util/logging.h"
#include "cavs/util/macros_gpu.h"

#include <string>
#include <initializer_list>
#include <unordered_map>

class Session {
 public:
  Session(int opt = 0, std::string name = "SimpleSession") {
    s_ = C_NewSession(name.c_str(), name.length(), opt);
  }

  void Run(std::vector<Sym> outputs,
      const std::initializer_list<std::pair<Sym&, void*>>& feed = {});
  void Run(Sym& output,
      const std::initializer_list<std::pair<Sym&, void*>>& feed = {}) {
    std::vector<Sym> out = {output};
    Run(out, feed);
  }

 private:
  C_Session* s_;
  std::unordered_map<string, C_Tensor*> feed_map_;
};

class MPISession : public Session {
 public:
  MPISession(int opt = 0, std::string name = "MPISession") : Session(opt, name) {}
  //int id;
};

#endif
