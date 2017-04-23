#ifndef CAVS_FRONTEND_CXX_SYM_H_
#define CAVS_FRONTEND_CXX_SYM_H_

#include "cavs/proto/op_def.pb.h"
#include "cavs/frontend/c_api.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <utility>

using std::string;
using std::vector;
using std::shared_ptr;
using std::ostream;
using std::pair;

class Sym {
 public:
  template <typename T> Sym (T constant);
  Sym& operator =(const Sym& sym);
  void Finalize(OpDef* op_def) const { return node_->Finalize(op_def); }

  //non-arguments operation
  static Sym Variable(C_Dtype type, const std::vector<int>& shape,
      const pair<string, OpDef::AttrDef>& filler = Ones(), string device = "GPU");
  static Sym Placeholder(C_Dtype type, const std::vector<int>& shape,
      string device = "GPU");
  static Sym MnistInput(int batch, string source, string file, string device = "GPU");
  static Sym Data(C_Dtype type, const std::vector<int>& shape, int batch,
      const pair<string, OpDef::AttrDef>& reader, string device = "GPU");
  static Sym DDV(C_Dtype type, const std::vector<int>& shape, int batch,
      const pair<string, OpDef::AttrDef>& filler = Ones(), string device = "GPU");
  //unary operation
  static Sym Abs(const Sym& a, string device = "GPU");
  static Sym Argmax(const Sym& a, int axis, string device = "GPU");
  static Sym Square(const Sym& a, string device = "GPU");
  static Sym Reduce_mean(const Sym& a, string device = "GPU");
  static Sym Reduce_sum(const Sym& a, string device = "GPU");
  static Sym Optimizer(const Sym& a);
  static Sym Optimizer(const Sym& a, vector<Sym> variables,
      float lr, int iters = 1, const string& projections = "");
  static Sym Maxpooling(const Sym&a, int HightWindow, int WidthWindow, string device = "GPU");
  static Sym Relu(const Sym&a, string device = "GPU");
  static Sym Flatten(const Sym& a);
  //binary operation
  static Sym Add(const Sym& a, const Sym& b, string device = "GPU");
  static Sym Sub(const Sym& a, const Sym& b, string device = "GPU");
  static Sym Mul(const Sym& a, const Sym& b, string device = "GPU");
  static Sym MatMul(const Sym& a, const Sym& b, string device = "GPU");
  static Sym FullyConnected(const Sym& a, const Sym& b, string device = "GPU");
  static Sym SoftmaxEntropyLogits(const Sym&a, const Sym& b, string device = "GPU");
  static Sym Equal(const Sym& a, const Sym& b, string device = "GPU");
  //ternary operation
  static Sym Conv(const Sym& a, const Sym& b, const Sym& c, string device = "GPU");
  //filler operation
  static std::pair<std::string, OpDef::AttrDef> Ones();
  static std::pair<std::string, OpDef::AttrDef> Zeros();
  static std::pair<std::string, OpDef::AttrDef> UniformRandom(int stride);
  static std::pair<std::string, OpDef::AttrDef> NormalRandom();
  static std::pair<std::string, OpDef::AttrDef> BinaryReader(const string& filename);
  //debug operations
  static void DumpGraph();
  void print();
  void* eval();
  ////////////////////////////////////////////////
  //unary operation
  Sym Abs() { return Abs(*this); }
  Sym Argmax(int axis) { return Argmax(*this, axis); };
  Sym Square() { return Square(*this); }
  Sym Reduce_mean() { return Reduce_mean(*this); };
  Sym Reduce_sum() { return Reduce_sum(*this); };
  Sym Optimizer() { return Optimizer(*this); }
  Sym Optimizer(vector<Sym> variables,
      float lr, int iters = 1, const string& projection = "") {
    return Optimizer(*this, variables, lr, iters, projection); 
  }
  Sym Maxpooling(int HightWindow, int WidthWindow) {
    return Maxpooling(*this, HightWindow, WidthWindow);
  }
  Sym Relu() { return Relu(*this); }
  Sym Flatten() { return Flatten(*this); };
  //binary operation
  Sym FullyConnected(const Sym& b) { return FullyConnected(*this, b); }
  Sym SoftmaxEntropyLogits(const Sym& b) { return SoftmaxEntropyLogits(*this, b); }
  //ternary operation
  Sym Conv(const Sym& b, const Sym& c) { return Conv(*this, b, c); }
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
    std::string label_;
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
      const string& label,
      const string& device,
      const std::vector<int>& shape = {},
      const vector<OpDef::AttrDef>& attrs = {});
  Sym(const string& op_name, const string& input,
      const vector<Sym>& variables = {},
      const float lr = 1,
      const int iters = 1,
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
