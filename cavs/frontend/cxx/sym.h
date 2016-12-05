#ifndef CAVS_FRONTEND_CXX_SYM_H_
#define CAVS_FRONTEND_CXX_SYM_H_

#include "cavs/midend/op_def.pb.h"
#include "cavs/frontend/c_api.h"
#include "cavs/frontend/cxx/chain.h"
#include "cavs/util/logging.h"

#include <string>
#include <initializer_list>
#include <vector>
#include <memory>
#include <iostream>

typedef std::initializer_list<int> Shape;
using std::string;
using std::vector;
using std::shared_ptr;
using std::ostream;

class Chain;
typedef struct SymBody{
  string op_name_;
  F_Dtype type_;
  Shape shape_;
  string output_;
  string device_;
  vector<const SymBody*> input_;
  Chain* chain_ = NULL;
  SymBody(); 
  void Finalize(cavs::OpDef* op_def) const;
  void* raw_data = NULL;
} SymBody;

class Sym {
 public:
  Sym() {}
  Sym& operator =(const Sym& sym);
  void Finalize(cavs::OpDef* op_def) const { body_->Finalize(op_def); }

  //non-arguments operation
  static Sym Variable(F_Dtype type, Shape shape, string output = "", string device = "GPU");
  static Sym Placeholder(F_Dtype type, Shape shape, string output = "", string device = "GPU");
  //unary operation
  static Sym Abs(const Sym& a, string output = "", string device = "GPU");
  //binary operation
  static Sym Add(const Sym& a, const Sym& b, string output = "", string device = "GPU");
  ////////////////////////////////////////////////
  //unary operation
  Sym Abs() { return Abs(*this); }
  //binary operation
  Sym Add(Sym& b) { return Add(*this, b); }
  ////////////////////////////////////////////////
  //operator overloading
  friend Sym operator +(const Sym& a, const Sym& b) { return Add(a, b); }
  void print();
        
  friend class Session;

 private:
  Sym(const string& op_name, const F_Dtype type, const Shape& shape, 
      const string& output, const string& device);
  inline string& op_name() const { return body_->op_name_; }
  inline F_Dtype type() const { return body_->type_; }
  inline Shape& shape() const { return body_->shape_; }
  inline string& output() const { return body_->output_; }
  inline string& device() const { return body_->device_; }
  void SetInput(const SymBody* sb) { body_->input_.push_back(sb); }
  shared_ptr<SymBody> body_;
};

#endif
