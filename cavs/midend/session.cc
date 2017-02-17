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

OpContext* SessionBase::GetContext(const OpDef& op_def) {
  OpContext* ctxt  = new OpContext();
  for (const string& input : op_def.input()) {
    //const Tensor* t = this->GetTensor(input); 
    ctxt->AppendInput(*(GetTensor(input)));
  }
  for (int i = 0; i < op_def.output_size(); i++) {
    const string& output = op_def.output(i);
    const Tensor* t = this->GetTensor(output);
    if (!t) {
      TensorShape shape(op_def.shape(i)); 
      Allocator* alloc = GetAllocator(op_def); 
      CHECK_NOTNULL(alloc);
      Tensor out(output, alloc, op_def.dtype(), std::move(shape));
      this->InsertTensor(out);
    }
    t = this->GetTensor(output);
    CHECK_NOTNULL(t);
    ctxt->AppendOutput(*t);
  }
  return ctxt;
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

  //LOG(INFO) << "here" << curr->op_def().DebugString();
  //LOG(INFO) << curr->inputs_size();
  //LOG(INFO) << isSource << accessed;
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
    LOG(INFO) << node->op_def().DebugString();
    LOG(INFO) << node->scope()->name();
  }
  for (auto* node : critical_path) {
    //LOG(INFO) << node->op_def().DebugString();
    //LOG(INFO) << node->scope()->name();
    LOG(INFO) << "compiling\t" << node->op_def().name();
    Statement* stmt = node->Compile(this);
    CHECK(stmt);
    executors_.push_back(stmt);
  }
  LOG(INFO) << "compile completed";

  return;
}

void SimpleSession::Run(const vector<string>& output_names,
    vector<Tensor>* output_tensors,
    const vector<string>& input_names,
    const vector<Tensor>& input_tensors) {
  Compile(output_names, input_names);
  FeedInput(input_names, input_tensors);
  //for (auto& one_pair : executors_) {
    //OpImpl* op = one_pair.first;
    //OpContext* context = one_pair.second;
    //op->Compute(context);
  //}
  for (auto* exe : executors_)
    exe->Run();
  FetchOutput(output_names, output_tensors);
}

void SimpleSession::FeedInput(const vector<string>& input_names,
    const vector<Tensor>& input_tensors) {
  CHECK(input_names.size() == input_tensors.size());
  for (int i = 0; i < input_names.size(); i++) {
    Tensor* t = &(tensor_map_[input_names[i]]);
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
    const Tensor& t = tensor_map_[output_names[i]];
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
