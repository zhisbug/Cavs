#ifndef CAVS_FRONTEND_CXX_SYM_H_
#define CAVS_FRONTEND_CXX_SYM_H_

#include "cavs/proto/op_def.pb.h"
#include "cavs/proto/func.pb.h"
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
  typedef pair<string, vector<OpDef::AttrDef>> ATTRIBUTE;
  enum MODE { STATIC_SYM = 1, DYNAMIC_SYM = 2 };
  inline static void SetMode(MODE mode) { mode_ = mode; }
  inline static void SetFuncname(string name) { func_name_ = name; }

  template <typename T> Sym (T constant);
  Sym& operator =(const Sym& sym);
  inline const OpDef& def() const { return node_->op_def; }
  inline OpDef* mutable_def() { return &(node_->op_def); }

  //non-arguments operation
  static Sym Variable(DataType type, const std::vector<int>& shape,
      const ATTRIBUTE& filler = Ones(), string device = "GPU");
  static Sym Placeholder(DataType type, const std::vector<int>& shape,
      string device = "GPU");
  static Sym MnistInput(int batch, string source, string file, string device = "GPU");
  static Sym Data(DataType type, const std::vector<int>& shape, int batch,
      const ATTRIBUTE& reader, string device = "GPU");
  static Sym DDV(DataType type, const std::vector<int>& shape, int batch,
      const ATTRIBUTE& filler = Ones(), string device = "GPU");
  //unary operation
  static Sym Abs(const Sym& a, string device = "GPU");
  static Sym Argmax(const Sym& a, int axis, string device = "GPU");
  static Sym Square(const Sym& a, string device = "GPU");
  static Sym Reduce_mean(const Sym& a, string device = "GPU");
  static Sym Reduce_sum(const Sym& a, string device = "GPU");
  static Sym Optimizer(const Sym& a);
  static Sym Optimizer(const Sym& a, vector<Sym> variables,
      float lr, float clip = 0.f, int iters = 1, const string& projections = "");
  static Sym Maxpooling(const Sym&a, int HightWindow, int WidthWindow, string device = "GPU");
  static Sym Relu(const Sym&a, string device = "GPU");
  static Sym Flatten(const Sym& a);
  //binary operation
  static Sym Add(const Sym& a, const Sym& b, string device = "GPU");
  static Sym Sub(const Sym& a, const Sym& b, string device = "GPU");
  static Sym Mul(const Sym& a, const Sym& b, string device = "GPU");
  static Sym MatMul(const Sym& a, const Sym& b, string device = "GPU");
  static Sym SoftmaxEntropyLogits(const Sym&a, const Sym& b, string device = "GPU");
  static Sym SoftmaxEntropyLoss(const Sym&a, const Sym& b, string device = "GPU");
  static Sym Equal(const Sym& a, const Sym& b, string device = "GPU");
  static Sym EmbeddingLookup(const Sym& a, const Sym& b, string device = "GPU");
  static Sym Reshape(const Sym& a, const std::vector<int>& shape);
  //ternary operation
  static Sym Conv(const Sym& a, const Sym& b, const Sym& c, string device = "GPU");
  static Sym FullyConnected(const Sym& x, const Sym& w, const Sym& b, string device = "GPU");
  //quaternary operation
  static Sym LSTM(const Sym& a, const Sym& b, int layer, int hidden, string device = "GPU");
  //filler operation
  static ATTRIBUTE Ones();
  static ATTRIBUTE Zeros();
  static ATTRIBUTE Const(float c);
  static ATTRIBUTE UniformNormalizer(int stride);
  static ATTRIBUTE Uniform(float minval, float maxval);
  static ATTRIBUTE Xavier();
  static ATTRIBUTE NormalRandom();
  static ATTRIBUTE BinaryReader(const string& filename);
  //debug operations
  static void DumpGraph();
  void print();
  void* eval();
  //////////////////////////////////////////////////////////////////////////////
  //unary operation
  Sym Abs() { return Abs(*this); }
  Sym Argmax(int axis) { return Argmax(*this, axis); };
  Sym Square() { return Square(*this); }
  Sym Reduce_mean() { return Reduce_mean(*this); };
  Sym Reduce_sum() { return Reduce_sum(*this); };
  Sym Optimizer() { return Optimizer(*this); }
  Sym Optimizer(vector<Sym> variables,
      float lr, float clip = 0.f, int iters = 1, const string& projection = "") {
    return Optimizer(*this, variables, lr, clip, iters, projection); 
  }
  Sym Maxpooling(int HightWindow, int WidthWindow) {
    return Maxpooling(*this, HightWindow, WidthWindow);
  }
  Sym Relu() { return Relu(*this); }
  Sym Flatten() { return Flatten(*this); }
  //binary operation
  Sym SoftmaxEntropyLogits(const Sym& b) { return SoftmaxEntropyLogits(*this, b); }
  Sym SoftmaxEntropyLoss(const Sym& b) { return SoftmaxEntropyLoss(*this, b); }
  Sym EmbeddingLookup(const Sym& b) { return EmbeddingLookup(*this, b); }
  Sym Reshape(const std::vector<int>& shape) { return Reshape(*this, shape); }
  //ternary operation
  Sym Conv(const Sym& b, const Sym& c) { return Conv(*this, b, c); }
  Sym FullyConnected(const Sym& w, const Sym& b) { return FullyConnected(*this, w, b); }
  //quaternary operation
  Sym LSTM(const Sym& b, int layer, int hidden) { return LSTM(*this, b, layer, hidden); }
  ////////////////////////////////////////////////
  //operator overloading
  friend Sym operator +(const Sym& a, const Sym& b) { return Add(a, b); }
  friend Sym operator -(const Sym& a, const Sym& b) { return Sub(a, b); }
  friend Sym operator *(const Sym& a, const Sym& b) { return Mul(a, b); }
        
  friend class Session;

 private:
  //typedef struct node {
    //string op_name_;
    //C_Dtype type_;
    //std::string label_;
    //std::vector<int> shape_;
    //string device_;
    //vector<string> output_;
    //vector<string> input_;
    //void Finalize(OpDef* op_def) const;
    //void* raw_data = NULL;
  //} node;
  typedef struct node_t {
    OpDef op_def;
    void* raw_data = NULL;
  } node_t;
    
  //Sym(const string& op_name,
      //const vector<string>& inputs, 
      //const C_Dtype type,
      //const string& label,
      //const string& device,
      //const std::vector<int>& shape = {},
      //const vector<OpDef::AttrDef>& attrs = {});
  //Sym(const string& op_name, const string& input,
      //const vector<Sym>& variables = {},
      //const float lr = 1,
      //const float clip = 0,
      //const int iters = 1,
      //const string& projections = "");
  inline std::vector<string> outputs() const { 
    std::vector<string> out;
    for (auto& o : def().output())
      out.push_back(o);
    return out;
  }
  inline const std::string& output(int idx) const { 
    return def().output(idx);
  }
  //inline string& op_name() const { return node_->op_name_; }
  inline const string& op_name() const { return def().name(); }
  //inline C_Dtype type() const { return node_->type_; }
  inline DataType type() const { return def().dtype(); }
  inline std::vector<int> shape(int idx) const {
    std::vector<int> out;
    for (auto& d : def().shape(idx).dim())
      out.push_back(d);
    return out;
  }
  inline DeviceType device() const { return def().device(); }
  shared_ptr<node_t> node_;
  static MODE mode_;
  static std::string func_name_;
  static std::unordered_map<std::string, FuncDef> func_def_;
};

#endif
