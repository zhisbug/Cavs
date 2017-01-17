#ifndef CAVS_FRONTEND_CXX_SYM_H_
#define CAVS_FRONTEND_CXX_SYM_H_

#include "cavs/proto/op_def.pb.h"
#include "cavs/frontend/c_api.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>
#include <memory>
#include <iostream>

typedef std::vector<int> Shape;
using std::string;
using std::vector;
using std::shared_ptr;
using std::ostream;

class Sym {
 public:
  Sym& operator =(const Sym& sym);
  void Finalize(OpDef* op_def) const { return node_->Finalize(op_def); }

  //non-arguments operation
  static Sym Variable(C_Dtype type, Shape shape, string output = "", string device = "GPU");
  static Sym Placeholder(C_Dtype type, Shape shape, string output = "", string device = "GPU");
  //unary operation
  static Sym Abs(const Sym& a, string output = "", string device = "GPU");
  static Sym Square(const Sym& a, string output = "", string device = "GPU");
  static Sym Optimizer(const Sym& a);
  //binary operation
  static Sym Add(const Sym& a, const Sym& b, string output = "", string device = "GPU");
  //debug operations
  static void DumpGraph();
  void print();
  ////////////////////////////////////////////////
  //unary operation
  Sym Abs() { return Abs(*this); }
  Sym Square() { return Square(*this); }
  Sym Optimizer() { return Optimizer(*this); }
  //binary operation
  Sym Square(Sym& b) { return Square(*this, b); }
  ////////////////////////////////////////////////
  //operator overloading
  friend Sym operator +(const Sym& a, const Sym& b) { return Add(a, b); }
        
  friend class Session;

 private:
  typedef struct node {
    string op_name_;
    C_Dtype type_;
    Shape shape_;
    string device_;
    vector<string> output_;
    vector<string> input_;
    void Finalize(OpDef* op_def) const;
    void* raw_data = NULL;
  } node;

  Sym(const string& op_name,
      const string& output, const vector<string>& inputs, 
      const C_Dtype type, const string& device, const Shape& shape);
  Sym(const string& op_name, const string& input);//for optimizer
  inline const std::vector<string>& outputs() const { 
    return node_->output_;
  }
  inline const std::string& output(int idx) const { 
    return node_->output_.at(idx);
  }
  inline string& op_name() const { return node_->op_name_; }
  inline C_Dtype type() const { return node_->type_; }
  inline Shape& shape() const { return node_->shape_; }
  inline string& device() const { return node_->device_; }
  shared_ptr<node> node_;
};

#endif
