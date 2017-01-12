#include "cavs/backend/op_decl.h"
#include "cavs/util/logging.h"

#include <unordered_map>

using std::unordered_map;
using std::string;
using std::vector;

namespace backend {

namespace op_factory {

typedef std::unordered_map<string, 
                           OpDeclRegister::Factory> OpDeclRegistry;
static OpDeclRegistry* GlobalOpDeclRegistry() {
  static OpDeclRegistry* global_op_decl_registry = new OpDeclRegistry();
  return global_op_decl_registry;
}
void OpDeclRegister::InitInternal(const string& name,
                              Factory factory) {
  GlobalOpDeclRegistry()->insert(std::make_pair(
      name, factory));
}

} //namespace op_factory

vector<OpDef> MakeGradient(const OpDef& def) {
  CHECK(op_factory::GlobalOpDeclRegistry()->count(def.name()) > 0);
  OpDecl *op_decl = op_factory::GlobalOpDeclRegistry()->at(def.name())(def);
  vector<OpDef> ret;
  op_decl->MakeGradient(&ret);
  delete(op_decl);
  return ret;
}

vector<TensorShapeDef> ShapeInference(const OpDef& def, 
    const vector<TensorShapeDef>& inputs) {
  
  CHECK(op_factory::GlobalOpDeclRegistry()->count(def.name()) > 0) << def.name();
  OpDecl *op_decl = op_factory::GlobalOpDeclRegistry()->at(def.name())(def);
  vector<TensorShapeDef> ret;
  op_decl->ShapeInference(&ret, inputs);
  delete(op_decl);
  return ret;

}

} //namespace backend
