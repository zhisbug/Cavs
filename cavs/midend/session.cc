#include "cavs/midend/session.h"
#include "cavs/midend/allocator.h"
#include "cavs/backend/op_def_builder.h"
#include "cavs/util/logging.h"

#include <unordered_map>
#include <list>

using std::string;
using std::vector;
using std::list;
using std::unordered_map;

namespace midend {

SimpleSession::SimpleSession(const DepGraph* graph)
    : SessionBase(graph), compiled_(false), round_(0) {}

void DepthSearch(const Node* curr,
    list<const Node*>* critical_path,
    unordered_map<const Node*, bool>* include) {
  bool isSource = (curr->inputs_size() == 0);
  bool accessed = (include->find(curr) != include->end());

  if (!accessed) {
    (*include)[curr] = true;
    if (!isSource) {
      for (auto* edge : curr->inputs()) {
        CHECK(edge->srcs_size() == 1 || edge->isStateful());
        //for (auto* node : edge->srcs()) {
        DepthSearch(edge->src(0), critical_path, include);
        //}
      }
    }
    critical_path->push_back(curr);
  }
  return;
}

void SimpleSession::Compile(
    const vector<string>& output_names, 
    const vector<string>& input_names) {
  list<const Node*> critical_path;
  unordered_map<const Node*, bool> include;
  for (auto& output : output_names) {
    const Node* node = graph_->FindNode(output);
    CHECK(node);
    DepthSearch(node, &critical_path, &include);
  }

  for (auto* node : critical_path) {
    //LOG(INFO) << node->op_def().DebugString();
    //LOG(INFO) << node->scope()->name();
    //LOG(INFO) << "compiling\t" << node->op_def().name();
    Statement* stmt = node->Compile(this);
    CHECK(stmt);
    executors_.push_back(stmt);
  }

  return;
}

void SimpleSession::Run(const vector<string>& output_names,
    vector<Tensor>* output_tensors,
    const vector<string>& input_names,
    const vector<Tensor>& input_tensors) {
  if (!compiled_) {
    Compile(output_names, input_names);
    compiled_ = true;
    round_ = 0;
  }
  //LOG(INFO) << "Feeding inputs...";
  FeedInput(input_names, input_tensors);
  //LOG(INFO) << "Executing...";
  for (auto* exe : executors_) {
    exe->SetRound(round_);
    exe->Run();
  }
  //LOG(INFO) << "Fetching output..";
  FetchOutput(output_names, output_tensors);
  //LOG(INFO) << "Execution completed";
  round_++;
}

void SimpleSession::FeedInput(const vector<string>& input_names,
    const vector<Tensor>& input_tensors) {
  CHECK(input_names.size() == input_tensors.size());
  for (int i = 0; i < input_names.size(); i++) {
    const Edge* edge = graph_->FindEdge(input_names[i]);
    CHECK(edge) << "Edge: " << input_names[i];
    //Tensor* t = &(tensor_map_[edge->scoped_name()]);
    Tensor* t = const_cast<Tensor*>(GetTensor(edge->scoped_name()));
    CHECK(t);
    if (t->device_type() == GPU)
      t->SyncWith(input_tensors[i]);
    else
      *t = input_tensors[i];
  }
}

void SimpleSession::FetchOutput(const vector<string>& output_names,
    vector<Tensor>* output_tensors) {
  CHECK(output_names.size() == output_tensors->size());
  for (int i = 0; i < output_names.size(); i++) {
    if (graph_->FindEdge(output_names[i])->isVirtual())
      continue;
    const Edge* edge = graph_->FindEdge(output_names[i]);
    CHECK(edge);
    //const Tensor& t = tensor_map_[edge->scoped_name()];
    const Tensor* t = GetTensor(edge->scoped_name());
    CHECK(t) << "Getting " << edge->scoped_name()
             << "\tin\n"   << DebugInfo();
    if (t->device_type() == GPU) {
      output_tensors->at(i).Rebase(GetAllocator(DeviceTypeToString(CPU)),
          *t);
      //DeviceContext::MemcpyDeviceToHost(&(output_tensors->at(i)), *t);
      output_tensors->at(i).SyncWith(*t);
      //LOG(INFO) << t->count();
      //LOG(INFO) << (output_tensors->at(i).data<float>())[0];
    }else {
      output_tensors->at(i) = *t;
    }
  }
}

void AddMPIOnPath(list<Node*>& critical_path) {
  auto iter = critical_path.begin(); 
  while (iter != critical_path.end()) {
    if ((*iter)->IsSingleNode()) {
      string name = (*iter)->output(0)->name();
      if (name.length() >= 13
          && name.substr(0, 13) == "variable_grad") {
        //we assume the output size of variable_grad node must equal 1
        CHECK((*iter)->outputs_size() == 1);
        OpDef comm;
        ::backend::OpDefBuilder("MPIAllGather")
          .Input(name)
          .Output(name)
          .Shape((*iter)->output(0)->shape())
          .Device("CPU")
          .Finalize(&comm);
        Node* comm_node = new Node(comm, (*iter)->scope());
        comm_node->AddInput((*iter)->output(0));
        comm_node->AddOutput((*iter)->output(0));
        critical_path.insert(++iter, comm_node);
      }
    }else if ((*iter)->IsScopedNode()) {
      AddMPIOnPath(static_cast<ScopedNode*>(*iter)->nodes_); 
      ++iter;
    }
  }
}

MPISession::MPISession(const DepGraph* graph)
    : SimpleSession(graph) {}

void MPISession::Compile(
    const vector<string>& output_names, 
    const vector<string>& input_names) {
  list<const Node*> critical_path;
  unordered_map<const Node*, bool> include;
  for (auto& output : output_names) {
    const Node* node = graph_->FindNode(output);
    CHECK(node);
    DepthSearch(node, &critical_path, &include);
  }

  for (auto* node : critical_path) {
    //Here, we assumpt the gradient of variables
    //should be communicated.
    //If the node generates a variable gradient,
    //it should be followed with a communication node.
    string name = node->output(0)->name();
    if (name.length() >= 13
        && name.substr(0, 13) == "variable_grad") {
      //we assume the output size of variable_grad node must equal 1
      CHECK(node->outputs_size() == 1);
      OpDef comm;
      ::backend::OpDefBuilder("MPIAllReduce")
        .Input(name)
        .Output(name)
        .Shape(node->output(0)->shape())
        .Device("CPU")
        .Finalize(&comm);
      Node* comm_node = new Node(comm, node->scope());
      comm_node->AddInput(node->output(0));
      comm_node->AddOutput(node->output(0));
    }
  }

  for (auto* node : critical_path) {
    LOG(INFO) << node->op_def().DebugString();
    LOG(INFO) << node->scope()->name();
    LOG(INFO) << "compiling\t" << node->op_def().name();
    Statement* stmt = node->Compile(this);
    CHECK(stmt);
    executors_.push_back(stmt);
  }

  return;
}

void MPISession::Run(const vector<string>& output_names,
    vector<Tensor>* output_tensors,
    const vector<string>& input_names,
    const vector<Tensor>& input_tensors) {
  SimpleSession::Run(output_names, output_tensors, input_names, input_tensors);
}


REGISTER_SESSION_BUILDER("SimpleSession", SimpleSession);
REGISTER_SESSION_BUILDER("MPISession", MPISession);

} //namespace midend
