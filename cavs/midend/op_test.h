#ifndef CAVS_MIDEND_OP_TEST_H_
#define CAVS_MIDEND_OP_TEST_H_

#include "cavs/midend/allocator.h"
#include "cavs/midend/types.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/session.h"
#include "cavs/midend/op.h"
#include "cavs/midend/tensor_test.h"
#include "cavs/util/logging.h"

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
      CHECK_NOTNULL(op_ = CreateOp(op_def_, sess_)); 
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

