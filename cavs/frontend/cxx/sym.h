#ifndef CAVS_FRONTEND_CXX_SYM_H_
#define CAVS_FRONTEND_CXX_SYM_H_

#include "cavs/proto/op_def.pb.h"
#include "cavs/proto/func_def.pb.h"
#include "cavs/frontend/c_api.h"
#include "cavs/util/logging.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>

using std::string;

class Sym {
 public:
  explicit Sym(const OpDef& op_def);
  Sym(const Sym& sym) { *this = sym; }
  Sym() : node_(nullptr) {}
  template <typename T>
  Sym (T constant);
  Sym& operator =(const Sym& sym);


  const string&        output(int idx) const;
  int                  output_size()   const;
  std::vector<string>  output()        const;
  std::vector<int>     shape(int idx)  const;
  inline DataType      type()          const { return def().dtype();      }
  inline DeviceType    device()        const { return def().device();     }
  inline const string& op_name()       const { return def().name();       }
  inline const OpDef&  def()           const { return node_->op_def;      }
  inline OpDef*        mutable_def()         { return &(node_->op_def);   }
  inline const void*   data()          const { return node_->raw_data;    }
  inline void**        mutable_data()        { return &(node_->raw_data); }

  typedef std::pair<string, std::vector<OpDef::AttrDef>> ATTRIBUTE;

  //non-arguments operation
  static Sym Variable(DataType type, const std::vector<int>& shape,
      const ATTRIBUTE& filler = Ones(), string device = "GPU");
  static Sym Placeholder(DataType type, const std::vector<int>& shape,
      string device = "GPU");
  static Sym Constant(DataType type, float value, const std::vector<int>& shape,
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
  static Sym Optimizer(const Sym& a, std::vector<Sym> variables,
      float lr, float clip = 0.f, int iters = 1, const string& projections = "");
  static Sym Maxpooling(const Sym& a, int HightWindow, int WidthWindow, string device = "GPU");
  static Sym Relu(const Sym& a, string device = "GPU");
  static Sym Sigmoid(const Sym& a, string device = "GPU");
  static Sym Tanh(const Sym& a, string device = "GPU");
  static Sym Flatten(const Sym& a);
  static Sym Slice(const Sym& a, int offset, int stride);
  //multi return value 
  static std::tuple<Sym, Sym, Sym> Split3(const Sym& a);
  static std::tuple<Sym, Sym, Sym, Sym> Split4(const Sym& a);
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
  static Sym Expand_dims(const Sym& a, int axis);
  //ternary operation
  static Sym Conv(const Sym& a, const Sym& b, const Sym& c, string device = "GPU");
  static Sym FullyConnected(const Sym& x, const Sym& w, const Sym& b, string device = "GPU");
  //quaternary operation
  static Sym LSTM(const Sym& a, const Sym& b, int layer, int hidden, string device = "GPU");
  //multi operators
  static Sym Concat(const std::vector<Sym>& syms, string device = "GPU");
  
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
  const  void* eval() const;
  //////////////////////////////////////////////////////////////////////////////
  //unary operation
  Sym Abs()            { return Abs(*this);          }
  Sym Argmax(int axis) { return Argmax(*this, axis); }
  Sym Square()         { return Square(*this);       }
  Sym Reduce_mean()    { return Reduce_mean(*this);  }
  Sym Reduce_sum()     { return Reduce_sum(*this);   }
  Sym Optimizer()      { return Optimizer(*this);    }
  Sym Optimizer(std::vector<Sym> variables,
      float lr, float clip = 0.f, int iters = 1, const string& projection = "") {
    return Optimizer(*this, variables, lr, clip, iters, projection); 
  }
  Sym Maxpooling(int HightWindow, int WidthWindow) {
    return Maxpooling(*this, HightWindow, WidthWindow);
  }
  Sym Relu()    { return Relu(*this);    }
  Sym Sigmoid() { return Sigmoid(*this); }
  Sym Tanh()    { return Tanh(*this);    }
  Sym Flatten() { return Flatten(*this); }
  Sym Slice(int offset, int stride) { return Slice(*this, offset, stride); }
  //multi return value 
  std::tuple<Sym, Sym, Sym> Split3()      { return Split3(*this); }
  std::tuple<Sym, Sym, Sym, Sym> Split4() { return Split4(*this); }
  //binary operation
  Sym SoftmaxEntropyLogits(const Sym& b)     { return SoftmaxEntropyLogits(*this, b); }
  Sym SoftmaxEntropyLoss(const Sym& b)       { return SoftmaxEntropyLoss(*this, b);   }
  Sym EmbeddingLookup(const Sym& b)          { return EmbeddingLookup(*this, b);      }
  Sym Reshape(const std::vector<int>& shape) { return Reshape(*this, shape);          }
  Sym Expand_dims(int axis)                  { return Expand_dims(*this, axis);    }
  //ternary operation
  Sym Conv(const Sym& b, const Sym& c)           { return Conv(*this, b, c);           }
  Sym FullyConnected(const Sym& w, const Sym& b) { return FullyConnected(*this, w, b); }
  //quaternary operation
  Sym LSTM(const Sym& b, int layer, int hidden)  { return LSTM(*this, b, layer, hidden); }
  ////////////////////////////////////////////////
  //operator overloading
  friend Sym operator +(const Sym& a, const Sym& b) { return Add(a, b); }
  friend Sym operator -(const Sym& a, const Sym& b) { return Sub(a, b); }
  friend Sym operator *(const Sym& a, const Sym& b) { return Mul(a, b); }

 private:
  typedef struct node_t {
    OpDef op_def;
    void* raw_data = NULL;
  } node_t;
    
  std::shared_ptr<node_t> node_;
};

inline std::vector<string> Sym::output() const { 
  std::vector<string> out;
  for (auto& o : def().output())
    out.push_back(o);
  return out;
}

inline const string& Sym::output(int idx) const { 
  CHECK(idx < def().output_size());
  return def().output(idx);
}

inline int Sym::output_size() const { 
  return def().output_size();
}

inline std::vector<int> Sym::shape(int idx) const { 
  std::vector<int> s;
  CHECK(idx < def().shape_size()) << idx << "\n" << def().DebugString();
  for (auto& d : def().shape(idx).dim())
    s.push_back(d);
  return s;
}

class FuncConf {
 public:
  inline static void FuncDefineBegin(const string& name) {
    Get()->name_ = name; 
    Get()->def_.set_name(name);
  }
  inline static FunctionDef FuncDefineEnd(const string& name) {
    CHECK(name == Get()->name_);
    FunctionDef ret = Get()->def_;
    Get()->name_.clear();
    Get()->def_.Clear();
    return ret;
  }
  inline static bool CheckInFunc() {
    return !(Get()->name_.empty());
  }
  inline static FunctionDef* mutable_funcdef() {
    CHECK(CheckInFunc());
    CHECK(!(Get()->def_.name().empty()));
    return &(Get()->def_); 
  }

 private:
  FuncConf() : name_("") {}
  static FuncConf* Get() { static FuncConf fc; return &fc; }
  string name_;
  FunctionDef def_;
};


#endif
