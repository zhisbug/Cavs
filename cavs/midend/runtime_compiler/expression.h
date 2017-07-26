#ifndef CAVS_MIDEND_RUNTIME_COMPILER_EXPRESSION_H_
#define CAVS_MIDEND_RUNTIME_COMPILER_EXPRESSION_H_

#include "cavs/midend/runtime_compiler/code_generator.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/proto/types.pb.h"

#include <string>

namespace midend {
namespace RTC {

class Base {
 public:
   virtual std::string toCode() const = 0;
};

class Expression : public Base {
 public:
  Expression(std::string op, DataType t) : op_(op), type_(t) {}
  virtual bool isAssignExpression() const { return false; }
  DataType dtype() const { return type_; }

 protected:
  std::string op_;
  DataType type_;
};

class UnaryExpression : public Expression {
 public:
  UnaryExpression(std::string op, std::string operand, DataType t)
    : Expression(op, t), operand_(operand) {}
  inline std::string toCode() const override {
    return op_ + "(" + operand_ + ")";
  }

 private:
  std::string operand_;
};

class BinaryExpression : public Expression {
 public:
  BinaryExpression(std::string op,
      std::string loperand, std::string roperand, DataType t)
    : Expression(op, t), loperand_(loperand), roperand_(roperand) {}
  inline std::string toCode() const override {
    return "(" + loperand_ + " " + op_ + " " + roperand_ + ")";
  }

 protected:
  std::string loperand_;
  std::string roperand_;
};

class AssignExpression : public BinaryExpression {
 public:
  AssignExpression(std::string op,
      std::string loperand, std::string roperand, DataType t)
    : BinaryExpression(op, loperand, roperand, t) {}
  bool isAssignExpression() const override { return true; }
  inline std::string toCode() const override {
    return loperand_ + " " + op_ + " " + roperand_;
  }
};

class Statement : public Base {
 public:
  Statement(DataType t) : type_(t) {}
  inline std::string toCode() const override {
    LOG(FATAL) << "Not implemented yet";
  }

 protected:
  DataType type_;
};

class VarDeclStatement : public Statement {
 public:
  VarDeclStatement(AssignExpression* ae) : ae_(ae), Statement(ae->dtype()) {}
  inline std::string toCode() const override {
    return CodeGenerator::typeToString(ae_->dtype()) + " " + ae_->toCode() + ";\n";
  }
 private:
  AssignExpression* ae_;
};

} //namespace RTC
} //namespace midend

#endif
