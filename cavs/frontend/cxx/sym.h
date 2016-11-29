#ifndef _SYM_H_
#define _SYM_H_

#include "cavs/core/op_def.pb.h"
#include "cavs/frontend/c_api.h"
#include "cavs/frontend/cxx/chain.h"

#include <string>
#include <initializer_list>
#include <vector>

typedef std::initializer_list<int> Shape;
using std::string;
using std::vector;

class Chain;
class Sym {
 public:
  Sym(string name, Dtype type, Shape shape, string device);
  string name() const { return name_; }
  Dtype type() const { return type_; }
  Shape shape() const { return shape_; }
  Sym* Input(int i) { return input_.at(i); }
  //Sym* Output(int i) { return output_.at(i); }
  void Finalize(cavs::OpDef* op_def) const;

  //non-arguments operation
  static Sym* Variable(Dtype type, Shape shape, string device = "GPU");
  static Sym* Placeholder(Dtype type, Shape shape, string device = "GPU");
  //unary operation
  static Sym* Abs(Sym* a, string device = "GPU");
  //binary operation
  static Sym* Add(Sym* a, Sym* b, string device = "GPU");

  ////////////////////////////////////////////////
  //unary operation
  Sym* Abs() { return Abs(this); }
  //binary operation
  Sym* Add(Sym* b) { return Add(this, b); }

 private:
  int id_;
  string name_;
  Dtype type_;
  Shape shape_;
  string device_;
  vector<Sym*> input_;
  //vector<Sym*> output_;
  static Chain* chain_;
};

#endif
