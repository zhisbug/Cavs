#include "cavs/midend/session.h"
#include "cavs/midend/allocator.h"
#include "cavs/midend/statement.h"
#include "cavs/backend/op_impl.h"
#include "cavs/backend/op_decl.h"
#include "cavs/util/logging.h"

#include <unordered_map>

using ::backend::OpImpl;
using ::backend::OpDecl;
using ::backend::CreateOp;

using std::string;
using std::vector;
using std::unordered_map;

namespace midend {

const Tensor* SessionBase::GetTensor(const string& name) const {
  if (tensor_map_.count(name) == 0)
    return NULL;
  else
    return &(tensor_map_.at(name));
}

void SessionBase::InsertTensor(const Tensor& t){
  CHECK(tensor_map_.count(t.name()) == 0);
  tensor_map_[t.name()] = t;
}

OpContext* SessionBase::GetContext(const Node* node) {
  OpContext* ctxt  = new OpContext();
  const OpDef& op_def = node->op_def();
  for (auto* input : node->inputs()) {
    const Tensor* t = this->GetTensor(input->scoped_name()); 
    CHECK(t) << "Getting " << input->scoped_name();
    ctxt->AppendInput(*t);
  }
  for (auto* output : node->outputs()) {
    const Tensor* t = this->GetTensor(output->scoped_name());
    if (!t) {
      //TensorShape shape(op_def.shape(i)); 
      TensorShape shape(output->shape()); 
      Allocator* alloc = GetAllocator(op_def); 
      CHECK_NOTNULL(alloc);
      LOG(INFO) << "allocating tensor for " << output->scoped_name()
                << " with shape info: " << shape.DebugInfo();
      Tensor out(output->scoped_name(), alloc, op_def.dtype(), std::move(shape));
      LOG(INFO) << out.DebugInfo();
      this->InsertTensor(out);
    }
    t = this->GetTensor(output->scoped_name());
    CHECK_NOTNULL(t);
    LOG(INFO) << t->DebugInfo();
    ctxt->AppendOutput(*t);
  }
  return ctxt;
}

string SessionBase::DebugInfo() {
  string ret;
  for (auto& one_pair : tensor_map_)
    ret += one_pair.first + "\t";
  return ret;
}

namespace session_factory {

typedef std::unordered_map<string, 
                           SessionRegister::Factory> SessionRegistry;
static SessionRegistry* GlobalSessionRegistry() {
  static SessionRegistry* global_session_registry 
    = new SessionRegistry();
  return global_session_registry;
}
void SessionRegister::InitInternal(
    const string& name, Factory factory) {
  GlobalSessionRegistry()->insert(std::make_pair(name, factory));
}

} //namespace session_factory

SessionBase* GetSession(const string& name, 
    const DepGraph* graph) {
  if (session_factory::GlobalSessionRegistry()->count(name) == 0)
    return NULL;
  else
    return session_factory::GlobalSessionRegistry()->at(name)(graph);
}

class SimpleSession : public SessionBase {
 public:
  SimpleSession(const DepGraph* graph);
  void Run(const vector<string>& output_names, 
           vector<Tensor>* output_tensors,
           const vector<string>& input_names,
           const vector<Tensor>& input_tensors) override;
 private:
  void FeedInput(const vector<string>& input_names,
                 const vector<Tensor>& input_tensors);
  void FetchOutput(const vector<string>& output_names,
                   vector<Tensor>* output_tensors);
  void Compile(const vector<string>& output_names, 
               const vector<string>& input_names);
  //std::vector<std::pair<OpImpl*, OpContext*>> executors_;
  std::vector<Statement*> executors_;
};

SimpleSession::SimpleSession(const DepGraph* graph)
    : SessionBase(graph) {
}

void DepthSearch(const Node* curr,
    vector<const Node*>* critical_path,
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
  vector<const Node*> critical_path;
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
  Compile(output_names, input_names);
  LOG(INFO) << "Feeding inputs...";
  FeedInput(input_names, input_tensors);
  //for (auto& one_pair : executors_) {
    //OpImpl* op = one_pair.first;
    //OpContext* context = one_pair.second;
    //op->Compute(context);
  //}
  LOG(INFO) << "Executing...";
  int i = 0;
  for (auto* exe : executors_) {
    exe->Run();
  }
  LOG(INFO) << "Fetching output..";
  FetchOutput(output_names, output_tensors);
}

void SimpleSession::FeedInput(const vector<string>& input_names,
    const vector<Tensor>& input_tensors) {
  CHECK(input_names.size() == input_tensors.size());
  for (int i = 0; i < input_names.size(); i++) {
    const Edge* edge = graph_->FindEdge(input_names[i]);
    CHECK(edge) << "Edge: " << input_names[i];
    Tensor* t = &(tensor_map_[edge->scoped_name()]);
    if (t->device_type() == GPU)
      DeviceContext::MemcpyHostToDevice(t, input_tensors[i]);
    else
      *t = input_tensors[i];
  }
}

void SimpleSession::FetchOutput(const vector<string>& output_names,
    vector<Tensor>* output_tensors) {
  CHECK(output_names.size() == output_tensors->size());
  for (int i = 0; i < output_names.size(); i++) {
    if (graph_->FindNode(output_names[i])->isVirtual())
      continue;
    const Edge* edge = graph_->FindEdge(output_names[i]);
    CHECK(edge);
    CHECK(tensor_map_.find(edge->scoped_name()) !=
          tensor_map_.end()) << "Getting " << edge->scoped_name()
          << "\tin\n" << DebugInfo();
    const Tensor& t = tensor_map_[edge->scoped_name()];
    if (t.device_type() == GPU) {
      (*output_tensors)[i].Rebase(GetAllocator(DeviceTypeToString(CPU)),
          t);
      DeviceContext::MemcpyDeviceToHost(&((*output_tensors)[i]), t);
    }else {
      (*output_tensors)[i] = t;
    }
  }
}

REGISTER_SESSION_BUILDER("SimpleSession", SimpleSession);

} //namespace midend
