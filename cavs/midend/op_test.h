#ifndef CAVS_MIDEND_OP_TEST_H_
#define CAVS_MIDEND_OP_TEST_H_

#include "cavs/midend/allocator.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/session_base.h"
#include "cavs/midend/tensor_test.h"
#include "cavs/backend/op_decl.h"
#include "cavs/backend/op_impl.h"
#include "cavs/util/types.h"

#include <string>

using ::backend::OpImpl;
using ::backend::CreateOp;
using ::backend::ShapeInference;

namespace midend {

namespace test{

class OpTest {
 public:
  OpTest (const OpDef& def);
  template <typename T>
  void AddTensorFromVector(const string& name,
                           const TensorShape& shape, 
                           const vector<T>& vals) {
    Edge* edge = main_scope()->FindEdge(name);
    CHECK_NOTNULL(edge);
    edge->SetShape(shape.to_def());

    Tensor input(edge->scoped_name(), GetAllocator(node_->op_def()), 
                 DataTypeToEnum<T>::value, shape); 
    sess_->InsertTensor(input);
    FillValues<T>(&input, vals);
  }

  template <typename T>
  void FetchTensor(const string& name,
                   vector<T>* data) {
    const Edge* edge = main_scope()->FindEdge(name);
    CHECK_NOTNULL(edge);
    FetchValues<T>(data, *(sess_->GetTensor(edge->scoped_name())));
  }

  bool RunTest () {
    const vector<TensorShapeDef>& input_shapes = node_->input_shapes();
    const vector<TensorShapeDef>& shape_def = ShapeInference(op_def_, input_shapes);
    CHECK(shape_def.size() == 1);
    node_->SetShape(shape_def);

    op_.reset(CreateOp(node_->op_def())); 
    context_.reset(sess_->GetContext(node_));
    op_->Compute(context_.get());
  }

 private:
  std::unique_ptr<OpImpl> op_;
  std::unique_ptr<OpContext> context_;
  OpDef op_def_;
  Node* node_;
  SessionBase* sess_;
};

OpTest ::OpTest(const OpDef& def) : op_def_(def) {
  //node_ = new Node(def, main_scope());
  //{
    //Edge* edge = new Edge(def.output(0), main_scope());
    //node_->AddOutput(edge);
    //edge->AddSource(node_);
    //edges_.push_back(edge);
    //edge->SetShape(def.shape(0));
  //}
  for (auto& i : def.input()) {
    Edge* edge = new Edge(i, main_scope());
  }

  node_ = main_scope()->AddOp(def);
  CHECK(node_);
  sess_ = new SessionBase();
}

} //namespace test

} //namespace midend

#endif
