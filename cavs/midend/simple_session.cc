#include "cavs/midend/simple_session.h"
#include "cavs/midend/allocator.h"
#include "cavs/util/logging.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/op_def_builder.h"

#include <iterator>

using std::string;
using std::vector;
using std::list;
using std::set;

namespace midend {

SimpleSession::SimpleSession() : SessionBase(), s_(main_scope()) {
 type_ = (int)SIMPLE;
}

void SimpleSession::DepthSearch(Node* curr,
    list<Node*>* critical_path,
    set<Node*>* include) {
  bool isSource = (curr->input_size() == 0);
  bool accessed = (include->find(curr) != include->end());

  if (!accessed) {
    //(*include)[curr] = true;
    //VLOG(V_DEBUG) << curr->debug_info();
    if (curr->IsScopedNode())
      include->insert(curr);
    if (!isSource) {
      for (auto* edge : curr->input()) {
        CHECK(edge->src_size() == 1 || edge->isVariable());
        //for (auto* node : edge->srcs()) {
        DepthSearch(const_cast<Node*>(edge->src(0)), critical_path, include);
        //}
      }
      for (auto* edge : curr->control_dependency()) {
        CHECK(edge->src_size() == 1);
        DepthSearch(const_cast<Node*>(edge->src(0)), critical_path, include);
        VLOG(V_DEBUG) << "[Dependency Info]:\n" << curr->debug_info()
                      << "\n==>\n"<< edge->src(0)->scoped_name();
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
  set<Node*> include;
  VLOG(V_DEBUG) << "Searching Critical Path";
  for (auto& output : output_names) {
    Node* node = const_cast<Node*>(s_->FindNode(output));
    CHECK(node);
    DepthSearch(node, &critical_path, &include);
  }
  CHECK(critical_path.size() >= 2);

  VLOG(V_DEBUG) << "============In Critical Path============";
  for (auto* node : critical_path) {
    VLOG(V_DEBUG) << "-------Node INFO\t"
                  << node->scope()->name() 
                  << ":" << node->name()
                  << "------";
    VLOG(V_DEBUG) << node->debug_info();
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
  VLOG(V_TIMING) << "Compiling for calculating the output ...";
  if (executors_.find(HashString(output_names)) == executors_.end()) {
    Compile(output_names);
  }
  VLOG(V_TIMING) << "Feeding inputs...";
  FeedInput(input_names, input_tensors);
  VLOG(V_TIMING) << "Executing...";
  for (auto* exe : executors_[HashString(output_names)]) {
    exe->Run();
  }
  VLOG(V_TIMING) << "Fetching output..";
  FetchOutput(output_names, output_tensors);
  VLOG(V_TIMING) << "Execution completed";
  Statement::IncRound();
}

void SimpleSession::FeedInput(const vector<string>& input_names,
    const vector<Tensor>& input_tensors) {
  CHECK(input_names.size() == input_tensors.size());
  for (int i = 0; i < input_names.size(); i++) {
    //const Edge* edge = graph_->FindEdge(input_names[i]);
    const Edge* edge = s_->FindEdge(input_names[i]);
    CHECK(edge) << "Edge: " << input_names[i];
    //Tensor* t = &(tensor_map_[edge->scoped_name()]);
    Tensor* t = const_cast<Tensor*>(GetTensor(edge->scoped_name()));
    CHECK(t) << input_names[i] << "\t" << debug_info();
    if (t->device_type() == GPU) {
      VLOG(V_DEBUG) << "Copying to GPU...";
      t->SyncWith(input_tensors[i]);
    }else {
      VLOG(V_DEBUG) << "Copying to CPU...";
      t->SyncWith(input_tensors[i]);
    }
  }
}

void SimpleSession::FetchOutput(const vector<string>& output_names,
    vector<Tensor>* output_tensors) {
  CHECK(output_names.size() == output_tensors->size());
  for (int i = 0; i < output_names.size(); i++) {
    VLOG(V_DEBUG) << "Fetching\t" << output_names[i]
                  << "\tVirtual?\t"
                  << s_->FindEdge(output_names[i])->isVirtual();
    //if (graph_->FindEdge(output_names[i])->isVirtual())
    if (s_->FindEdge(output_names[i])->isVirtual())
      continue;
    //const Edge* edge = graph_->FindEdge(output_names[i]);
    const Edge* edge = s_->FindEdge(output_names[i]);
    CHECK_NOTNULL(edge);
    const Tensor* t = GetTensor(edge->scoped_name());
    CHECK(t) << "Getting " << edge->scoped_name()
             << "\tin\n"   << debug_info();
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
