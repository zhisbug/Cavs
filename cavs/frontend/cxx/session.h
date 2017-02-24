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
  Session(std::string name = "SimpleSession") {
    s_ = C_NewSessionWithDG(
        name.c_str(), name.length(), C_GetDefaultDG());
  }

  void Run(const std::initializer_list<Sym>& outputs,
      const std::initializer_list<std::pair<Sym&, void*>>& feed = {});
  inline void Run(const Sym& output,
      const std::initializer_list<std::pair<Sym&, void*>>& feed = {}) {
    Run({output}, feed);
  }

 private:
  C_Session* s_;
  std::unordered_map<string, C_Tensor*> feed_map_;
};

#endif
