#ifndef CAVS_CORE_OP_TEST_H_
#define CAVS_CORE_OP_TEST_H_

#include "cavs/core/allocator.h"
#include "cavs/core/types.h"
#include "cavs/core/logging.h"
#include "cavs/core/tensor.h"
#include "cavs/core/session.h"
#include "cavs/core/op.h"
#include "cavs/core/tensor_test.h"

#include <string>

namespace cavs{

namespace test{

class OpTestBase {
 public:
  OpTestBase(const OpDef& def);
  template <typename T>
  void AddTensorFromVector(const string& name,
                           const TensorShape shape, 
                           const vector<T> vals) {
    Tensor* input = new Tensor(name, alloc_, DataTypeToEnum<T>::value, shape); 
    sess_->InsertTensor(input);
    FillValues<T>(input, vals);
  }
  template <typename T>
  void FetchTensor(const string& name,
                   vector<T>& data) {
    FetchValues<T>(data, sess_->GetTensor(name));
  }

  bool RunTest () {
      CHECK(op_ = CreateOp(op_def_, sess_)); 
      op_->Compute();
  }

 private:
  Session* sess_;
  Op* op_;
  OpDef op_def_;
  Allocator* alloc_;
};

OpTestBase::OpTestBase(const OpDef& def) : op_(NULL){
  op_def_.CopyFrom(def); 
  alloc_ = GetAllocator(op_def_);
  sess_ = new Session();
}

} //namespace test

} //namespace cavs
#endif

