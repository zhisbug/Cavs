#ifndef SESSION_H_
#define SESSION_H_

#include "tensor.h"

class Session {
 public:
  bool Session::InsertTensor(string name, Tensor* t);
  bool Session::CreateAndInsertTensor(t);
  Tensor* GetTensor(string name);
 private:
  unordered_map<string, Tensor*> tensor_map;
};

Session* simple_session();
#endif
