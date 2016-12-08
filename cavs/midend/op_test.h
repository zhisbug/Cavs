#ifndef CAVS_MIDEND_OP_TEST_H_
#define CAVS_MIDEND_OP_TEST_H_

#include "cavs/midend/allocator.h"
#include "cavs/midend/types.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/op.h"
#include "cavs/midend/session.h"
#include "cavs/midend/tensor_test.h"

#include <string>

namespace cavs{

namespace test{

class SessionOpTest : public SessionBase {
};

class OpTest {
 public:
  OpTest (const OpDef& def);
  template <typename T>
  void AddTensorFromVector(const string& name,
                           const TensorShape& shape, 
                           const vector<T>& vals) {
    Tensor input(name, GetAllocator(op_def_), 
                 DataTypeToEnum<T>::value, shape); 
    sess_->InsertTensor(input);
    FillValues<T>(&input, vals);
  }
  template <typename T>
  void FetchTensor(const string& name,
                   vector<T>* data) {
    FetchValues<T>(data, *(sess_->GetTensor(name)));
  }
  bool RunTest () {
    op_.reset(CreateOp(op_def_)); 
    context_.reset(new OpContext(op_def_, sess_));
    op_->Compute(context_.get());
  }

 private:
  std::unique_ptr<Op> op_;
  std::unique_ptr<OpContext> context_;
  OpDef op_def_;
  SessionOpTest* sess_;
};

OpTest ::OpTest(const OpDef& def) {
  op_def_.CopyFrom(def); 
  sess_ = new SessionOpTest();
}

} //namespace test

} //namespace cavs

#endif
