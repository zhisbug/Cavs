#include "cavs/midend/simple_session.h"
#include "cavs/midend/statement.h"
#include "cavs/backend/op_impl_mpi_functor.h"
#include "cavs/util/op_def_builder.h"

#include <mpi.h>
#include <unordered_map>
#include <list>

using std::string;
using std::vector;
using std::list;
using std::unordered_map;

using ::backend::MPIAllReduceFunctor;

namespace midend {

class MPISession: public SimpleSession {
 public:
  MPISession(const DepGraph* graph);
  ~MPISession();
  void Run(const vector<string>& output_names, 
           vector<Tensor>* output_tensors,
           const vector<string>& input_names,
           const vector<Tensor>& input_tensors) override;
  int SessionType() override { return MPI; }
 private:
  void Compile(const vector<string>& output_names) override;
  void FetchOutput(const vector<string>& output_names,
                   vector<Tensor>* output_tensors) override;
};

void AddMPIOnPath(list<Node*>& critical_path) {
  auto iter = critical_path.begin(); 
  while (iter != critical_path.end()) {
    if ((*iter)->IsSingleNode()) {
      string name = (*iter)->output(0)->name();
      LOG(INFO) << name;
      if ((name.length() >= 13 && name.substr(0, 8) == "Variable")
          && name.substr(name.length()-5, 5) == "_grad") {
        if ((*iter)->op_def().name() == "MatMul") {
          LOG(INFO) << "SFB mechanism ENABLing...";
          CHECK((*iter)->outputs_size() == 1);
          CHECK((*iter)->inputs_size() == 2);
          OpDef comm;
          OpDefBuilder("MPISFB")
            .Input((*iter)->input(0)->name())
            .Input((*iter)->input(1)->name())
            .Output((*iter)->output(0)->name())
            .Shape((*iter)->output(0)->shape())
            .Attr((*iter)->op_def())
            .Device("GPU")
            .Finalize(&comm);
          Node* comm_node = new SingleNode(comm, (*iter)->scope());
          comm_node->AddInput((*iter)->input(0));
          comm_node->AddInput((*iter)->input(1));
          comm_node->AddOutput((*iter)->output(0));
          *iter = comm_node;
          //sleep(3);
        }else {
          //we assume the output size of variable_grad node must equal 1
          CHECK((*iter)->outputs_size() == 1);
          OpDef comm;
          OpDefBuilder("MPIAllReduce")
            .Input(name)
            .Output(name)
            .Shape((*iter)->output(0)->shape())
            .Device("GPU")
            .Finalize(&comm);
          Node* comm_node = new SingleNode(comm, (*iter)->scope());
          comm_node->AddInput((*iter)->output(0));
          comm_node->AddOutput((*iter)->output(0));
          critical_path.insert(++iter, comm_node);
          continue;
        }
      }
    }else if ((*iter)->IsScopedNode()) {
      AddMPIOnPath(static_cast<ScopedNode*>(const_cast<Node*>(*iter))->nodes_); 
    }
    iter++;
  }
}

MPISession::MPISession(const DepGraph* graph)
    : SimpleSession(graph){
  MPI_Init(NULL, NULL);
}

MPISession::~MPISession() {
  MPI_Finalize();
}

void MPISession::Compile(
    const vector<string>& output_names) {
  list<Node*> critical_path;
  unordered_map<Node*, bool> include;
  for (auto& output : output_names) {
    Node* node = const_cast<Node*>(graph_->FindNode(output));
    CHECK(node);
    DepthSearch(node, &critical_path, &include);
  }

  //Here, we assumpt the gradient of variables
  //should be communicated.
  //If the node generates a variable gradient,
  //it should be followed with a communication node.
  AddMPIOnPath(critical_path);

  CHECK(executors_.find(HashString(output_names)) == executors_.end());
  vector<Statement*>* executor = &executors_[HashString(output_names)];
  for (auto* node : critical_path) {
    LOG(INFO) << node->op_def().DebugString();
    LOG(INFO) << node->scope()->name();
    LOG(INFO) << "compiling\t" << node->op_def().name();
    Statement* stmt = node->Compile(this);
    CHECK(stmt);
    executor->push_back(stmt);
  }
  return;
}

void MPISession::Run(const vector<string>& output_names,
    vector<Tensor>* output_tensors,
    const vector<string>& input_names,
    const vector<Tensor>& input_tensors) {
  SimpleSession::Run(output_names, output_tensors, input_names, input_tensors);
  //LOG(INFO) << "round: " << round_;
}

void MPISession::FetchOutput(const vector<string>& output_names,
    vector<Tensor>* output_tensors) {
  SimpleSession::FetchOutput(output_names, output_tensors);
  for (auto& t : *output_tensors) {
    if (!t.Empty()) {
      MPIAllReduceFunctor<float>::Compute(t.data<float>(),
          t.mutable_data<float>(), t.count());
    }
  }
}

REGISTER_SESSION_BUILDER("MPISession", MPISession);

} //namespace midend

