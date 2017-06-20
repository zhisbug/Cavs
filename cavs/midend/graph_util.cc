#include "cavs/midend/graph_util.h"
#include "cavs/midend/statement.h"
#include "cavs/midend/statement_builder.h"
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

//void DeduceAndApplyOneGradNode(
    //Scope* s,
    //const Node* node,
    //const string& edge) {
  //const vector<OpDef>& grads = 
    //::backend::MakeGradient(node->op_def()); 
  //CHECK(grads.size()) << node->op_def().DebugString();
  //bool exist = false;
  //for (auto& grad : grads) {
    //if (std::find(grad.output().begin(), grad.output().end(),
         //GetGradientName(edge)) == grad.output().end()) {
      //continue;
    //}
    //CHECK(!exist) << "Two grad possible?"
                  //<< node->op_def().DebugString()
                  //<< grad.DebugString();
    ////Node* grad_node = s->AddNode(grad);
    //Node* grad_node = s->AddOp(grad);
    //if (grad_node) {
      //const vector<TensorShapeDef>& inputs = 
        //grad_node->input_shapes();
      //const vector<TensorShapeDef>& shapes = 
        //::backend::ShapeInference(grad, inputs);
      //grad_node->SetShape(shapes);
    //}
    //exist = true;
  //}
  //CHECK(exist) << "No gradient wrt the edge name";
  //return;
//}

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


//void GroupClosedSet(
    //const vector<string>& vars,
    //const Edge* loss,
    //const string& solver,
    //const float lr,
    //const float clip,
    //const string& proj,
    //Scope* loss_scope) {
  //unordered_map<const Node*, bool> recalculate;
  //for (auto& var_name : vars) {
    //VLOG(V_DEBUG) << var_name;
    //const Edge* var = loss_scope->FindEdge(var_name);
    //list<const Node*> newly_traversed;
    //TraverseCriticalPath(loss_scope, loss, var,
        //&recalculate, &newly_traversed);
  //}

  //if (clip > 0) {
    //vector<string> outputs;
    //vector<TensorShapeDef> outputs_shape;
    //for (auto& var_name : vars) {
      //outputs.emplace_back(GetGradientName(var_name));
      //const Edge* var = loss_scope->FindEdge(var_name);
      //outputs_shape.emplace_back(var->shape());
    //}
    //OpDef clipper;  
    //OpDefBuilder("Clip")
      //.Input(outputs)
      //.Output(outputs)
      //.Shape(outputs_shape)
      //.Device("GPU")
      //.AttrSingle<float>("clip", clip)
      //.Finalize(&clipper);
    ////loss_scope->AddNode(clipper);
    //loss_scope->AddOp(clipper);
  //}

  //for (auto& var_name : vars) {
    //const Edge* var = loss_scope->FindEdge(var_name);
    //OpDef update;  
    //OpDefBuilder(solver)
      //.Input(var_name)
      //.Input(GetGradientName(var_name))
      //.Output(var_name)
      //.Shape(var->shape())
      //.AttrSingle<float>("Learning_rate", lr)
      //.Device("GPU")
      //.Finalize(&update);
    ////loss_scope->AddNode(update);
    //loss_scope->AddOp(update);

    //if (proj.length() > 0) {
      //OpDef projection;  
      //OpDefBuilder(proj)
        //.Input(var_name)
        //.Output(var_name)
        //.Shape(var->shape())
        //.Device("GPU")
        //.Finalize(&projection);
      ////loss_scope->AddNode(projection);
      //loss_scope->AddOp(projection);
    //}
  //}
//}

OpDef GraphUtil::PartialGrad(const Node* node, const string& edge) {
  const vector<OpDef>& grads = 
    ::backend::MakeGradient(node->op_def()); 
  CHECK(grads.size()) << node->op_def().DebugString();

  for (auto& grad : grads) {
    if (std::find(grad.output().begin(), grad.output().end(),
         GetGradientName(edge)) != grad.output().end()) {
      return grad;
    }
  }

  LOG(FATAL) << "No gradient valid!";
}

bool GraphUtil::GenCriticalPath(vector<bool>* cpath,
    vector<unordered_map<size_t, OpDef>>* grads,
    const Edge* curr,
    const Edge* loss) {
  CHECK(curr->src_size(true) == 1) << curr->DebugInfo();
  CHECK(curr->src_size(false) == 1 || curr->isStateful()) << curr->DebugInfo();
  LOG_IF(INFO, curr->dst_size() > 1) << curr->DebugInfo();
  VLOG(V_DEBUG) << "GenCriticalPath:\t" << curr->DebugInfo();
  if (curr == loss) {
    CHECK(node2idx_.find(curr->src(0)) != node2idx_.end());
    int idx = node2idx_[curr->src(0)];
    CHECK(cpath->size() > idx);
    cpath->at(idx) = true;
    return true;
  }else {
    bool inpath = false;
    for (Node* node : curr->dst(true)) {
      CHECK(node2idx_.find(curr->src(0)) != node2idx_.end());
      int idx = node2idx_[node];
      CHECK(cpath->size() > idx);
      if (!cpath->at(idx)) {
        for (int i = 0; i < node->output_size(); i++) {
          Edge* edge = node->output(i);
          if (GenCriticalPath(cpath, grads, edge, loss)) {
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
        size_t hashcode = GetHash(def);
        if (grads->at(idx).find(hashcode) != grads->at(idx).end()) {
          grads->at(idx).emplace(hashcode, def);
        }
        inpath = true;
      }
    }
    return inpath;
  }
}

void GraphUtil::GenGradient(Scope* loss_scope,
    const vector<bool>& critical_path,
    const vector<unordered_map<size_t, OpDef>>& grads) {
  CHECK(critical_path.size() == grads.size());
  CHECK(critical_path.size() == s_->typological_sorted_nodes_.size());
  VLOG(V_DEBUG) << "Forwarding...";
  for (int i = 0; i < critical_path.size(); i++) {
    if (critical_path[i]) {
      loss_scope->AddOp(s_->typological_sorted_nodes_[i]->op_def());
      CHECK(!grads[i].empty());
    }else {
      CHECK(grads[i].empty());
    }
  }

  VLOG(V_DEBUG) << "Backwarding...";
  for (int i = critical_path.size()-1 ; i >= 0; i--) {
    if (critical_path[i]) {
      VLOG(V_DEBUG) << "Backwarding for "
                    << s_->typological_sorted_nodes_[i]->op_def().DebugString();
      for (auto& iter : grads[i]) {
        VLOG(V_DEBUG) << "Adding grad op\n" << iter.second.DebugString();
        Node* grad_node = loss_scope->AddOp(iter.second);
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

void GraphUtil::ComputeGradient(
    Scope* loss_scope,
    const vector<string>& vars,
    const Edge* loss,
    const Scope* main_scope) {
  CHECK(loss_scope);
  CHECK(main_scope);
  vector<bool> critical_path(main_scope->typological_sorted_nodes_.size(), false);
  vector<unordered_map<size_t, OpDef>> grads(main_scope->typological_sorted_nodes_.size());
  for (auto& var_name : vars) {
    VLOG(V_DEBUG) << var_name;
    const Edge* var = loss_scope->FindEdge(var_name);
    GenCriticalPath(&critical_path, &grads, var, loss);
  }
  VLOG(V_DEBUG) << "Generating gradient...";
  GenGradient(loss_scope, critical_path, grads);
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
  //loss_scope->AddNode(clipper);
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

GraphUtil::GraphUtil(Scope* s) : s_(s) {
  for (int i = 0; i < s->typological_sorted_nodes_.size(); i++) {
    node2idx_[s->typological_sorted_nodes_[i]] = i; 
  }
}

Node* GraphUtil::AddOptimizerOp(const OpDef& def) {
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

  OpDef const_op;
  OpDefBuilder("ConstOp")
    .Output(GetGradientName(loss))
    .Shape(loss_edge->shape())
    .AttrSingle("init", 1.f)
    .Device("GPU")
    .Finalize(&const_op);
  loss_scope->AddOp(const_op);

  VLOG(V_DEBUG) << "Compute Gradients...";
  //GroupClosedSet(var_names, loss_edge, solver, lr, clip, proj, loss_scope);
  ComputeGradient(loss_scope, var_names, loss_edge, s_);
  VLOG(V_DEBUG) << "Gradient process...";
  if (clip > 0) GradientProcess(loss_scope, var_names, clip);
  ApplyGradient(loss_scope, var_names, solver, proj, lr);
  ScopedNode* sn = new ScopedNode(s_, loss_scope, def, iters);

  return sn;
}

TensorShapeDef GraphUtil::AddFunction(const FunctionDef& def) {
  string func_scope_name = def.name();
  Scope* func_scope = new Scope(s_, func_scope_name);

  TensorShapeDef out_shape;
  bool push_op = false;
  for (auto& op : def.ops()) {
    Node* node = func_scope->AddOp(op);
    const vector<TensorShapeDef>& input_shapes = 
      node->input_shapes();
    const vector<TensorShapeDef>& shape_def = 
      ::backend::ShapeInference(op, input_shapes);
    node->SetShape(shape_def);
    if (op.name() == "Push") {
      //Currently, push only has one output.
      //There is one and only one push op per function
      CHECK(op.output_size() == 1);
      CHECK(!push_op);
      out_shape = shape_def[0];
      push_op = true;
    }
  }
  CHECK(push_op);

  return out_shape;
}

} //namespace midend
