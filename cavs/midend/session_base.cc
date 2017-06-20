#include "cavs/midend/session_base.h"
#include "cavs/midend/allocator.h"
#include "cavs/util/logging.h"
//#include "cavs/util/op_util.h"

#include <unordered_map>

using std::string;
using std::unordered_map;

namespace midend {

const Tensor* SessionBase::GetTensor(
    const string& name, bool recursive) const {
  if (!recursive) {
    return (tensor_map_.find(name) == tensor_map_.end()) ?
            NULL : &(tensor_map_.at(name));
  }else {
    CHECK(name.find_last_of(":") != string::npos);
    string tensor_name = name.substr(name.find_last_of(":")+1);
    string scope_name  = name.substr(0, name.find_last_of(":"));
    while (tensor_map_.find(scope_name+":"+tensor_name) == tensor_map_.end()
        && scope_name.find_last_of(":") != string::npos) {
      scope_name = scope_name.substr(0, scope_name.find_last_of(":")); 
    }
    return tensor_map_.find(scope_name+":"+tensor_name) == tensor_map_.end() ?
           NULL : &(tensor_map_.at(scope_name+":"+tensor_name));
  }
}

void SessionBase::InsertTensor(const Tensor& t){
  CHECK(tensor_map_.count(t.name()) == 0);
  tensor_map_[t.name()] = t;
}

OpContext* SessionBase::GetContext(const Node* node) {
  OpContext* ctxt  = new OpContext();
  const OpDef& op_def = node->op_def();
  for (auto* input : node->input()) {
    const Tensor* t = GetTensor(input->scoped_name()); 
    CHECK(t) << "Getting " << input->scoped_name();
    ctxt->AppendInput(*t);
  }
  for (auto* output : node->output()) {
    const Tensor* t = GetTensor(output->scoped_name());
    if (!t) {
      //LOG(INFO) << output->name() << "???";
      //LOG(INFO) << (GetTensor(output->name()) == NULL) << "???";
      const Tensor* upper_t = GetTensor(output->scoped_name(), true);
      if (upper_t) {
        LOG(INFO) << "Found underlying tensor(" << upper_t->name()
                  << "," << upper_t->count() << " elements"
                  << ") for " << output->scoped_name()
                  << " with shape info: " << output->shape().DebugString();
        Tensor out(output->scoped_name(), *upper_t);
        InsertTensor(out);
      }else if (GetSingleArg<bool>(op_def, "ShareMemory", false)) {
        //currently, we only support sharing memory
        //for single-input and single-output operators
        //and only share output(0) with input(0)
        //CHECK(node->inputs_size() == 1); //reshape need two inputs
        CHECK(node->output_size() == 1); 
        Tensor out(output->scoped_name(),
            *GetTensor(node->input(0)->scoped_name()));
        out.Reshape(output->shape());
        LOG(INFO) << "Share Memory Tensor" << out.DebugInfo();
        InsertTensor(out);
      }else {
        CHECK(output->shape().dim_size() > 0);
        TensorShape shape(output->shape()); 
        Allocator* alloc = GetAllocator(op_def); 
        CHECK_NOTNULL(alloc);
        VLOG(V_DEBUG) << "allocating tensor for " << output->scoped_name()
                  << " with shape info: " << shape.DebugInfo();
        Tensor out(output->scoped_name(), alloc, op_def.dtype(), std::move(shape));
        VLOG(V_DEBUG) << out.DebugInfo();
        InsertTensor(out);
      }
    }
    t = GetTensor(output->scoped_name());
    CHECK(t) << t->DebugInfo();
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

SessionBase* GetSession(const string& name) {
    //const DepGraph* graph) {
  if (session_factory::GlobalSessionRegistry()->count(name) == 0)
    return NULL;
  else
    //return session_factory::GlobalSessionRegistry()->at(name)(graph);
    return session_factory::GlobalSessionRegistry()->at(name)();
}

} //namespace midend
