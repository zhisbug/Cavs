#ifndef _SYM_H_
#define _SYM_H_

#include "cavs/core/op_def.pb.h"

#include <initializer_list>

typedef std::initializer_list<int> Shape;

class Sym {
 public:
  Sym(string name, Dtype type, Shape shape, string device) 
    : name_(name), type_(type), shape_(shape), device_(device) {}
  string name() const { return name_; }
  Dtype type() const { return type_; }
  Shape shape() const { return type_; }
  OpDef& Finalize();
  Sym* Input(i) { return input_.at(i); }
  Sym* Output(i) { return output_.at(i); }

  //non-arguments operation
  static Sym* Variable(Dtype type, Shape shape);
  static Sym* Placeholder(Dtype type, Shape shape);
  //unary operation
  static Sym* Abs(Sym* a);
  //binary operation
  static Sym* Add(Sym* a, Sym* b);

  ////////////////////////////////////////////////
  //unary operation
  Sym* Abs();
  //binary operation
  Sym* Add(Sym* b);

 private:
  unique_ptr<OpDef> op_def_;
  string name_;
  Dtype type_;
  Shape shape_;
  string device_;
  vector<Sym*> input_;
  vector<Sym*> output_;
  static Chain* chain_;

};

#endif
