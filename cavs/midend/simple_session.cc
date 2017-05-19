#include "cavs/midend/simple_session.h"
#include "cavs/midend/allocator.h"
#include "cavs/backend/op_def_builder.h"
#include "cavs/util/logging.h"
#include "cavs/util/macros_gpu.h"

#include <iterator>

using std::string;
using std::vector;
using std::list;
using std::unordered_map;

namespace midend {

SimpleSession::SimpleSession(const DepGraph* graph)
    : SessionBase(graph), round_(0) {}

void SimpleSession::DepthSearch(Node* curr,
    list<Node*>* critical_path,
    unordered_map<Node*, bool>* include) {
  bool isSource = (curr->inputs_size() == 0);
  bool accessed = (include->find(curr) != include->end());

  if (!accessed) {
    (*include)[curr] = true;
    if (!isSource) {
      for (auto* edge : curr->inputs()) {
        CHECK(edge->srcs_size() == 1 || edge->isStateful());
        //for (auto* node : edge->srcs()) {
        DepthSearch(const_cast<Node*>(edge->src(0)), critical_path, include);
        //}
      }
    }
    critical_path->push_back(curr);
  }
  return;
}

string SimpleSession::HashString(const vector<string>& input) {
  string str;
  for (auto& s : input)
    str += s;
  return str;
}

void SimpleSession::Compile(
    const vector<string>& output_names) {
  list<Node*> critical_path;
  unordered_map<Node*, bool> include;
  for (auto& output : output_names) {
    Node* node = const_cast<Node*>(graph_->FindNode(output));
    CHECK(node);
    DepthSearch(node, &critical_path, &include);
  }

  VLOG(V_DEBUG) << "============In Critical Path============";
  for (auto* node : critical_path) {
    VLOG(V_DEBUG) << "-------Node INFO\t"
                  << node->scope()->name() 
                  << ":" << node->op_def().name()
                  << "------";
    VLOG(V_DEBUG) << node->op_def().DebugString();
  }
  VLOG(V_DEBUG) << "============End Critical Path============";

  CHECK(executors_.find(HashString(output_names)) == executors_.end());
  vector<Statement*>* executor = &executors_[HashString(output_names)];
  for (auto* node : critical_path) {
    Statement* stmt = node->Compile(this);
    CHECK(stmt);
    executor->push_back(stmt);
  }

  return;
}

void SimpleSession::Run(const vector<string>& output_names,
    vector<Tensor>* output_tensors,
    const vector<string>& input_names,
    const vector<Tensor>& input_tensors) {
  if (executors_.find(HashString(output_names)) == executors_.end()) {
    Compile(output_names);
  }
  VLOG(V_TIMING) << "Feeding inputs...";
  FeedInput(input_names, input_tensors);
  VLOG(V_TIMING) << "Executing...";
  for (auto* exe : executors_[HashString(output_names)]) {
    exe->SetRound(round_);
    exe->Run();
  }
  VLOG(V_TIMING) << "Fetching output..";
  FetchOutput(output_names, output_tensors);
  VLOG(V_TIMING) << "Execution completed";
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
    VLOG(V_DEBUG) << "Fetching" << output_names[i];
    if (graph_->FindEdge(output_names[i])->isVirtual())
      continue;
    const Edge* edge = graph_->FindEdge(output_names[i]);
    CHECK(edge);
    const Tensor* t = GetTensor(edge->scoped_name());
    CHECK(t) << "Getting " << edge->scoped_name()
             << "\tin\n"   << DebugInfo();
    if (t->device_type() == GPU) {
      output_tensors->at(i).Rebase(GetAllocator(DeviceTypeToString(CPU)),
          *t);
      output_tensors->at(i).SyncWith(*t);
    }else {
      output_tensors->at(i) = *t;
    }
  }
}

REGISTER_SESSION_BUILDER("SimpleSession", SimpleSession);

} //namespace midend
