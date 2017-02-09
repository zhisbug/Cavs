#ifndef CAVS_FRONTEND_CXX_SYM_H_
#define CAVS_FRONTEND_CXX_SYM_H_

#include "cavs/proto/op_def.pb.h"
#include "cavs/frontend/c_api.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>
#include <memory>
#include <iostream>

using std::string;
using std::vector;
using std::shared_ptr;
using std::ostream;

class Sym {
 public:
  Sym& operator =(const Sym& sym);
  void Finalize(OpDef* op_def) const { return node_->Finalize(op_def); }

  //non-arguments operation
  static Sym Variable(C_Dtype type, std::vector<int> shape,
      const OpDef::AttrDef& attr = Ones(), string device = "GPU");
  static Sym Placeholder(C_Dtype type, std::vector<int> shape,
      string device = "GPU");
  //unary operation
  static Sym Abs(const Sym& a, string device = "GPU");
  static Sym Square(const Sym& a, string device = "GPU");
  static Sym Optimizer(const Sym& a);
  static Sym Optimizer(const Sym& a, vector<Sym> variables,
      int iters = 0, const string& projections = "");
  //binary operation
  static Sym Add(const Sym& a, const Sym& b, string device = "GPU");
  static Sym Sub(const Sym& a, const Sym& b, string device = "GPU");
  static Sym Mul(const Sym& a, const Sym& b, string device = "GPU");
  //filler operation
  static OpDef::AttrDef Ones();
  //debug operations
  static void DumpGraph();
  void print();
  ////////////////////////////////////////////////
  //unary operation
  Sym Abs() { return Abs(*this); }
  Sym Square() { return Square(*this); }
  Sym Optimizer() { return Optimizer(*this); }
  Sym Optimizer(vector<Sym> variables, int iters = 0,
      const string& projection = "") {
    return Optimizer(*this, variables, iters, projection); 
  }
  //binary operation
  ////////////////////////////////////////////////
  //operator overloading
  friend Sym operator +(const Sym& a, const Sym& b) { return Add(a, b); }
  friend Sym operator -(const Sym& a, const Sym& b) { return Sub(a, b); }
  friend Sym operator *(const Sym& a, const Sym& b) { return Mul(a, b); }
        
  friend class Session;

 private:
  typedef struct node {
    string op_name_;
    C_Dtype type_;
    std::vector<int> shape_;
    string device_;
    vector<string> output_;
    vector<string> input_;
    void Finalize(OpDef* op_def) const;
    void* raw_data = NULL;
  } node;

  Sym(const string& op_name,
      const vector<string>& inputs, 
      const C_Dtype type,
      const string& device,
      const std::vector<int>& shape,
      const OpDef::AttrDef& attr = OpDef::AttrDef());
  Sym(const string& op_name, const string& input,
      const vector<Sym>& variables = {},
      const int iters = 0,
      const string& projections = "");
  inline const std::vector<string>& outputs() const { 
    return node_->output_;
  }
  inline const std::string& output(int idx) const { 
    return node_->output_.at(idx);
  }
  inline string& op_name() const { return node_->op_name_; }
  inline C_Dtype type() const { return node_->type_; }
  inline std::vector<int> shape() const { return node_->shape_; }
  inline string& device() const { return node_->device_; }
  shared_ptr<node> node_;
};

#endif
