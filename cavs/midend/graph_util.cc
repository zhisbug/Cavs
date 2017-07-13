#include "cavs/midend/graph_util.h"
#include "cavs/midend/statement.h"
#include "cavs/backend/op_decl.h"
#include "cavs/util/logging.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/op_def_builder.h"
#include "cavs/util/op_util.h"

#include <string>
#include <algorithm>
#include <list>

using namespace std;

namespace midend {

//bool TraverseCriticalPath(Scope* loss_scope,
      //const Edge* loss, const Edge* curr,
      //unordered_map<const Node*, bool>* fwd_path,
      //list<const Node*>* newly_traversed) {
  //CHECK(curr->srcs_size() == 1) << curr->DebugInfo();
  //LOG_IF(INFO, curr->dsts_size() > 1) << curr->DebugInfo();
  ////LOG(INFO) << curr->DebugInfo();
  //if (curr == loss) {
    //for (auto* node : *newly_traversed) {
      ////loss_scope->AddNode(node->op_def());
      //loss_scope->AddOp(node->op_def());
    //}
    //newly_traversed->clear();
    //return true;
  //}
  //const Node* node = curr->dst(0);
  //CHECK(node);
  //if (fwd_path->find(node) != fwd_path->end() && (*fwd_path)[node]) {
    //DeduceAndApplyOneGradNode(loss_scope, node, curr->name());
    //for (auto* node : *newly_traversed) {
      ////loss_scope->AddNode(node->op_def());
      //loss_scope->AddOp(node->op_def());
    //}
    //newly_traversed->clear();
    //return true;
  //}
  ////CHECK(curr->dsts_size() == 1) << curr->dsts_size();
  //if (fwd_path->find(node) == fwd_path->end() ||
      //!(*fwd_path)[node]) {
    //(*fwd_path)[node] = false;
    //newly_traversed->push_back(node);
    //bool in_path = false;
    //for (auto* edge : node->outputs()) {
      //if (TraverseCriticalPath(loss_scope, loss, edge,
            //fwd_path, newly_traversed)) {
        //const vector<OpDef>& grads = 
          //::backend::MakeGradient(node->op_def()); 
        //CHECK(grads.size()) << node->op_def().DebugString();
        //DeduceAndApplyOneGradNode(loss_scope, node, curr->name());
        //in_path = true;
        //(*fwd_path)[node] = true; 
      //}
    //}
    //if (!in_path) {
      //CHECK(newly_traversed->size() > 0);
      //newly_traversed->pop_back();
    //}
  //}
  //return (*fwd_path)[node];
//}

OpDef GraphUtil::PartialGrad(const Node* node, const string& edge) {
  CHECK(node->IsSingleNode());
  const vector<OpDef>& grads = 
    ::backend::MakeGradient(dynamic_cast<const SingleNode*>(node)->op_def()); 
  CHECK(grads.size()) << dynamic_cast<const SingleNode*>(node)->op_def().DebugString();

  for (auto& grad : grads) {
    if (std::find(grad.output().begin(), grad.output().end(),
         GetGradientName(edge)) != grad.output().end()) {
      return grad;
    }
  }
  LOG(FATAL) << "No gradient valid for " << edge
             << "\n from " << node->debug_info();
}

bool GraphUtil::GenCriticalPath(vector<bool>* cpath,
    vector<unordered_map<size_t, OpDef>>* grads,
    const Edge* curr,
    const Edge* loss,
    const Scope* scope) {
  CHECK(curr->scope() == scope);
  CHECK(loss->scope() == scope);
  CHECK(curr->src_size(true) == 1) << curr->debug_info();
  CHECK(curr->src_size(false) == 1 || curr->isVariable()) << curr->debug_info();
  LOG_IF(INFO, curr->dst_size() > 1) << curr->scoped_name();
  VLOG(V_DEBUG) << "GenCriticalPath:\t" << curr->scoped_name();
  CHECK(scope->node2idx_.find(curr->src(0)) != scope->node2idx_.end());
  if (curr == loss) {
    int idx = scope->node2idx_.at(curr->src(0));
    CHECK(cpath->size() > idx);
    cpath->at(idx) = true;
    return true;
  }else {
    bool inpath = false;
    for (Node* node : curr->dst(true)) {
      int idx = scope->node2idx_.at(node);
      CHECK(cpath->size() > idx);
      if (!cpath->at(idx)) {
        for (int i = 0; i < node->output_size(); i++) {
          Edge* edge = node->output(i);
          if (GenCriticalPath(cpath, grads, edge, loss, scope)) {
            cpath->at(idx) = true;
            inpath = true;
          }
        }
        if (cpath->at(idx)) {
          const OpDef& def = PartialGrad(node, curr->name());
          size_t hashcode = GetHash(def);
          CHECK(grads->at(idx).find(hashcode) == grads->at(idx).end());
          grads->at(idx).emplace(hashcode, def);
        }
      }else {
        const OpDef& def = PartialGrad(node, curr->name());
        //VLOG(V_DEBUG) << "CHECKING whether this partial is already inserted\t"
                      //<< def.DebugString();
        size_t hashcode = GetHash(def);
        if (grads->at(idx).find(hashcode) == grads->at(idx).end()) {
          //VLOG(V_DEBUG) << "CHECKING RESULT: False\n";
          grads->at(idx).emplace(hashcode, def);
        }else {
          //VLOG(V_DEBUG) << "CHECKING RESULT: True\n";
        }
        inpath = true;
      }
    }
    return inpath;
  }
}

void GraphUtil::GenGradient(Scope* loss_scope,
    const vector<bool>& critical_path,
    const vector<unordered_map<size_t, OpDef>>& grads,
    const Edge* loss_edge) {
  CHECK(critical_path.size() == grads.size());
  CHECK(critical_path.size() == s_->typological_sorted_nodes_.size());
  VLOG(V_DEBUG) << "Forwarding...";
  unordered_map<int, GraphNode*> gnode_map;
  for (int i = 0; i < critical_path.size(); i++) {
    if (critical_path[i]) {
      VLOG(V_DEBUG) << "[" << i << "]:\n"
                    << dynamic_cast<SingleNode*>(s_->typological_sorted_nodes_[i])->op_def().DebugString();
      CHECK(s_->typological_sorted_nodes_[i]->IsSingleNode());
      SingleNode* node = loss_scope->AddOp(dynamic_cast<SingleNode*>(s_->typological_sorted_nodes_[i])->op_def());
      if (node->IsGraphOp()) {
        gnode_map[i] = dynamic_cast<GraphNode*>(node);
      }
      //we have to loose this constraint because of dependency support
      //the dependent node does not need to backward.
      //CHECK(!grads[i].empty());
    }else {
      CHECK(grads[i].empty());
    }
  }

  OpDef const_op;
  OpDefBuilder("ConstOp")
    .Output(GetGradientName(loss_edge->name()))
    .Shape(loss_edge->shape())
    .AttrSingle("init", 1.f)
    .Device("GPU")
    .Finalize(&const_op);
  loss_scope->AddOp(const_op);

  VLOG(V_DEBUG) << "Backwarding...";
  for (int i = critical_path.size()-1 ; i >= 0; i--) {
    if (critical_path[i]) {
      CHECK(s_->typological_sorted_nodes_[i]->IsSingleNode());
      VLOG(V_DEBUG) << "Backwarding for "
                    << dynamic_cast<SingleNode*>(s_->typological_sorted_nodes_[i])->op_def().DebugString();
      SingleNode* grad_node = NULL;
      for (auto& iter : grads[i]) {
        VLOG(V_DEBUG) << "Adding grad op\n" << iter.second.DebugString();
        grad_node = loss_scope->AddOp(iter.second);
        CHECK(grad_node);
        VLOG(V_DEBUG) << "Getting input shape...";
        const vector<TensorShapeDef>& inputs = 
          grad_node->input_shapes();
        VLOG(V_DEBUG) << "Shaping Inference...";
        const vector<TensorShapeDef>& shapes = 
          ::backend::ShapeInference(iter.second, inputs);
        VLOG(V_DEBUG) << "Setting shape...";
        grad_node->SetShape(shapes);
        VLOG(V_DEBUG) << "One grad added";
      }

      if (dynamic_cast<SingleNode*>(s_->typological_sorted_nodes_[i])->IsGraphOp()) {
        CHECK_NOTNULL(grad_node);
        CHECK(gnode_map.find(i) != gnode_map.end());
        dynamic_cast<GraphGradNode*>(grad_node)->SetGraphForwardNode(gnode_map[i]);
        for (auto&& func_name : {"Node"}) {
          //find the childscope of father or ancestor(optimizer case)
          VLOG(V_DEBUG) << "Compute Gradient for " << func_name << "...";
          const Scope* func_scope = s_->FindChildScope(func_name);
          Scope* func_grad_scope = new Scope(func_scope, GetGradientName(func_name));
          CHECK(func_scope);
          CHECK(func_grad_scope);
          ComputeGradientForFunction(func_grad_scope, func_scope);
        }
      }
    }
  }
}

void GraphUtil::ComputeGradient(
    Scope* loss_scope,
    ScopedNode* loss_scope_node,
    const vector<string>& vars,
    const Edge* loss_edge,
    const Scope* main_scope) {
  CHECK(loss_scope);
  CHECK(main_scope);
  vector<bool> critical_path(main_scope->typological_sorted_nodes_.size(), false);
  vector<unordered_map<size_t, OpDef>> grads(main_scope->typological_sorted_nodes_.size());
  for (auto& var_name : vars) {
    VLOG(V_DEBUG) << var_name;
    const Edge* var = loss_scope->FindEdge(var_name);
    if (!GenCriticalPath(&critical_path, &grads, var, loss_edge, main_scope)) {
      LOG(FATAL) << var_name << "\tis not a trainable variable";
    }
  }

  //dealing with the dependency
  for (int i = 0; i < critical_path.size(); i++) {
    if (critical_path[i]) {//other nodes are dependent on this node
      for (auto* e : main_scope->typological_sorted_nodes_[i]->control_dependency()) {
        CHECK(e->scope() == main_scope);
        CHECK(e->src_size(true) == 1);
        for (auto* n : e->src(true)) {
          if (!critical_path[main_scope->node2idx_.at(n)]) {
            VLOG(V_DEBUG) << "ScopedNode dependency: "
                          << loss_scope_node->scoped_name()
                          << " depends on " << e->scoped_name();
            loss_scope_node->AddControlDependency(e);
          }
        }
      }

      for (auto* e : main_scope->typological_sorted_nodes_[i]->output()) {
        for (auto* n : e->control_dependency()) {
          CHECK(n->scope() == main_scope);
          if (!critical_path[main_scope->node2idx_.at(n)]) {
            VLOG(V_DEBUG) << "Triger dependency in ScopedNode: "
                          << n->scoped_name()
                          << " depends on " << e->scoped_name();
            critical_path[main_scope->node2idx_.at(n)] = true;
          }
        }
      }
    }
  }

  VLOG(V_DEBUG) << "Generating gradient...";
  GenGradient(loss_scope, critical_path, grads, loss_edge);
}

void GraphUtil::GradientProcess(
    Scope* loss_scope,
    const vector<string>& vars,
    float clip) {
  vector<string> outputs;
  vector<TensorShapeDef> outputs_shape;
  for (auto& var_name : vars) {
    outputs.emplace_back(GetGradientName(var_name));
    const Edge* var = loss_scope->FindEdge(var_name);
    outputs_shape.emplace_back(var->shape());
  }
  OpDef clipper;  
  OpDefBuilder("Clip")
    .Input(outputs)
    .Output(outputs)
    .Shape(outputs_shape)
    .Device("GPU")
    .AttrSingle<float>("clip", clip)
    .Finalize(&clipper);
  loss_scope->AddOp(clipper);
}

void GraphUtil::ApplyGradient(
    Scope* loss_scope,
    const vector<string>& vars,
    const string& solver,
    const string& proj,
    float lr) {
  for (auto& var_name : vars) {
    const Edge* var = loss_scope->FindEdge(var_name);
    OpDef update;  
    OpDefBuilder(solver)
      .Input(var_name)
      .Input(GetGradientName(var_name))
      .Output(var_name)
      .Shape(var->shape())
      .AttrSingle<float>("Learning_rate", lr)
      .Device("GPU")
      .Finalize(&update);
    loss_scope->AddOp(update);

    if (proj.length() > 0) {
      OpDef projection;  
      OpDefBuilder(proj)
        .Input(var_name)
        .Output(var_name)
        .Shape(var->shape())
        .Device("GPU")
        .Finalize(&projection);
      loss_scope->AddOp(projection);
    }
  }
}

GraphUtil::GraphUtil(Scope* s) : s_(s) {}

ScopedNode* GraphUtil::AddOptimizerOp(const OpDef& def) {
  CHECK(def.input_size() == 1);
  const string& loss = def.input(0);
  vector<string> var_names = GetListArg<string>(def, string("Vars"));
  int    iters  = GetSingleArg(def, "Iters"        , 0         );
  float  lr     = GetSingleArg(def, "Learning_rate", 0.f       );
  float  clip   = GetSingleArg(def, "Clip"         , 0.f       );
  string proj   = GetSingleArg(def, "Projection"   , string(""));
  string solver = GetSingleArg(def, "Solver"       , string(""));

  CHECK(!var_names.empty());
  CHECK(iters > 0);
  CHECK(lr > 0);
  CHECK(clip >= 0);
  CHECK(!solver.empty());

  Scope* loss_scope = new Scope(s_, def.output(0));

  const Edge* loss_edge = s_->FindEdge(loss);
  CHECK(loss_edge);

  ScopedNode* sn = new ScopedNode(s_, def.output(0), iters);
  VLOG(V_DEBUG) << "Compute Gradients...";
  ComputeGradient(loss_scope, sn, var_names, loss_edge, s_);
  VLOG(V_DEBUG) << "Gradient process...";
  if (clip > 0) GradientProcess(loss_scope, var_names, clip);
  ApplyGradient(loss_scope, var_names, solver, proj, lr);

  sn->SetContainedScope(loss_scope);

  return sn;
}

TensorShapeDef GraphUtil::AddFunction(const FunctionDef& def) {
  string func_scope_name = def.name();
  Scope* func_scope = new Scope(s_, func_scope_name);

  TensorShapeDef out_shape;
  bool push_op = false;
  for (auto& op : def.ops()) {
    SingleNode* node = func_scope->AddOp(op);
    const vector<TensorShapeDef>& input_shapes = 
      node->input_shapes();
    const vector<TensorShapeDef>& shape_def = 
      ::backend::ShapeInference(op, input_shapes);
    node->SetShape(shape_def);
    if (node->name() == "Push") {
      //Currently, push only has one output.
      //There is one and only one push op per function
      CHECK(node->output_size() == 1);
      CHECK(!push_op);
      push_op = true;
      out_shape = shape_def[0];
    }
  }
  CHECK(push_op);

  return out_shape;
}

void GraphUtil::ComputeGradientForFunction(
    Scope* func_grad_scope,
    const Scope* func_scope) {
  CHECK(func_grad_scope);
  CHECK(func_scope);
  vector<bool> critical_path(func_scope->typological_sorted_nodes_.size(), false);
  vector<unordered_map<size_t, OpDef>> grads(func_scope->typological_sorted_nodes_.size());
  vector<Edge*> origins;
  vector<Edge*> terminals;
  for (auto* node : func_scope->typological_sorted_nodes_) {
    VLOG(V_DEBUG) << node->debug_info();
    if (node->name() == "Gather") {
      CHECK(func_scope->node2idx_.find(node) != func_scope->node2idx_.end());
      critical_path[func_scope->node2idx_.at(node)] = true;
      //Gather node is special, because it is a source node without input
      //so PartialGrad can not be applied to the Gather node differentiation
      const vector<OpDef>& grad_defs = 
        ::backend::MakeGradient(dynamic_cast<const SingleNode*>(node)->op_def()); 
      CHECK(grad_defs.size() == 1);
      grads.at(func_scope->node2idx_.at(node)).emplace(GetHash(grad_defs[0]), grad_defs[0]);
      CHECK(node->output_size() == 1);
      origins.push_back(node->output(0));
    }
    //for each optimizer/functionGrad, there must be a starting point
    //it is either a constant op(init = 1) or a buffer fetcher op
    if (node->name() == "Push") {
      OpDef fetchUpperGradOp;
      OpDefBuilder("FetchUpperGrad")
        .Output(GetGradientName(node->output(0)->name()))
        .Shape(node->output(0)->shape())
        .Device("GPU")
        .Finalize(&fetchUpperGradOp);
      func_grad_scope->AddOp(fetchUpperGradOp);
      CHECK(node->output_size() == 1);
      terminals.push_back(node->output(0));
    }
    if (node->name() == "Scatter") {
      CHECK(node->output_size() == 1);
      terminals.push_back(node->output(0));
    }
  }
  CHECK(terminals.size() == 2) << terminals.size();

  for (auto* o_edge : origins) {
    for (auto* t_edge : terminals) {
      vector<bool> cpath_each(critical_path.size(), false);
      vector<unordered_map<size_t, OpDef>> grads_each(grads.size());
      if (!GenCriticalPath(&cpath_each, &grads_each, o_edge, t_edge, func_scope)) {
        LOG(FATAL) << o_edge->name()
                   << "\tis not a trainable variable in function";
      }
      for (int i = 0; i < critical_path.size(); i++) {
        critical_path[i] = critical_path[i] || cpath_each[i];
        grads[i].insert(grads_each[i].begin(), grads_each[i].end());
      }
    }
  }

  //for (int i = 0; i < critical_path.size(); i++) {
    //if (critical_path[i]) {
      //VLOG(V_DEBUG) << i << "\tth operator:";
      //VLOG(V_DEBUG) << func_scope->typological_sorted_nodes_[i]->debug_info();
    //}
  //}

  VLOG(V_DEBUG) << "Generating gradient...";
  GenGradientForFunction(func_grad_scope, critical_path, grads, func_scope);
}

void GraphUtil::GenGradientForFunction(Scope* func_grad_scope,
    const vector<bool>& critical_path,
    const vector<unordered_map<size_t, OpDef>>& grads,
    const Scope* func_scope) {
  CHECK(critical_path.size() == grads.size());
  CHECK(critical_path.size() == func_scope->typological_sorted_nodes_.size());
  VLOG(V_DEBUG) << "Function auto-diff does not need forwarding...";

  VLOG(V_DEBUG) << "Function auto-diff backwarding...";
  for (int i = critical_path.size()-1 ; i >= 0; i--) {
    if (critical_path[i]) {
      VLOG(V_DEBUG) << "Backwarding for "
                    << func_scope->typological_sorted_nodes_[i]->debug_info();
      for (auto& iter : grads[i]) {
        VLOG(V_DEBUG) << "Adding grad op\n" << iter.second.DebugString();
        SingleNode* grad_node = func_grad_scope->AddOp(iter.second);
        CHECK(grad_node);
        VLOG(V_DEBUG) << "Getting input shape...";
        const vector<TensorShapeDef>& inputs = 
          grad_node->input_shapes();
        VLOG(V_DEBUG) << "Shaping Inference...";
        const vector<TensorShapeDef>& shapes = 
          ::backend::ShapeInference(iter.second, inputs);
        VLOG(V_DEBUG) << "Setting shape...";
        grad_node->SetShape(shapes);
        VLOG(V_DEBUG) << "One grad added";
      }
    }
  }
}

} //namespace midend
