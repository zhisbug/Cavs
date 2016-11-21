#ifndef CAVS_CORE_SESSION_H_
#define CAVS_CORE_SESSION_H_

#include "cavs/core/tensor.h"

namespace cavs {

class Session {
 public:
  const Tensor* GetTensor(const string& name) const;
  bool CreateTensor(const string& name);
  bool InsertTensor(Tensor* t);
 private:
  unordered_map<string, Tensor*> tensor_map;
};

Session* simple_session();

} //namespace cavs

#endif
