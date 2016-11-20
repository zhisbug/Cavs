#ifndef OP_TEST_H_
#define OP_TEST_H_

#include <string>

#include "allocator.h"
#include "types.h"

namespace cavs{

namespace test{

class OpTestBase {
 public:
  OpTestBase() : device_type_("GPU"), alloc_(gpu_allocator()), op_(NULL) {}
  void SetOpDef(const OpDef& op_def) { op_def_.CopyFrom(op_def); }
  bool InitOp() {
    op_ = CreateOp(op_def_, tensor_map); 
    return (op_ == NULL)? false : true;
  }

  template <typename T>
  void AddInputFromVector(const string input_name;
                          const TensorShape& shape, 
                          const vector<T>& data) {
    Tensor* input = new Tensor(input_name, alloc_, DataTypeToEnum<T>.value, shape); 
    sess->InsertTensor(input_name, input);
    FillValues(input, data);
  }

  bool RunTest () {
      op_->compute();
  }

  template <typename T>
  void FetchOutput(const string input_name;
                   const TensorShape& shape, 
                   const vector<T>& data) {
    FetchValues(data, input);
  }

 private:
  Session *sess;
  Op* op_;
  OpDef op_def_;
  string device_type_;
  Allocator* alloc_;
};

} //namespace test

} //namespace cavs
#endif

