#ifndef CAVS_MIDEND_RUNTIME_COMPILER_EXPRESSION_H_
#define CAVS_MIDEND_RUNTIME_COMPILER_EXPRESSION_H_

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
  Expression(std::string op) : op_(op) {}
  virtual bool isAssignExpression() const { return false; }

 private:
  std::string op_;
};

class UnaryExpression : public Expression {
 public:
  UnaryExpression(std::string op, std::string operand)
    : Expression(op), operand_(operand) {}
  std::string toCode() const override {
    LOG(FATAL) << "Not implemented yet";
  }

 private:
  std::string operand_;
};

class BinaryExpression : public Expression {
 public:
  BinaryExpression(std::string op, std::string loperand, std::string roperand)
    : Expression(op), loperand_(loperand), roperand_(roperand) {}
  std::string toCode() const override {
    LOG(FATAL) << "Not implemented yet";
  }

 private:
  std::string loperand_;
  std::string roperand_;
};

class AssignExpression : public BinaryExpression {
 public:
  AssignExpression(std::string op, std::string loperand, std::string roperand)
    : BinaryExpression(op, loperand, roperand) {}
  bool isAssignExpression() const override { return true; }
  std::string toCode() const override {
    LOG(FATAL) << "Not implemented yet";
  }
};

class Statement : public Base {
 public:
  std::string toCode() const override {
    LOG(FATAL) << "Not implemented yet";
  }
};

class ExprStatement : public Statement {
 public:
  ExprStatement(AssignExpression* ae) : ae_(ae) {}
  std::string toCode() const override {
    LOG(FATAL) << "Not implemented yet";
  }
 private:
  AssignExpression* ae_;
};

class VarDeclStatement: public Statement {
 public:
  VarDeclStatement(DataType t, AssignExpression* e)
    : type_(t), ae_(e) {}
  std::string toCode() const override {
    LOG(FATAL) << "Not implemented yet";
  }
 private:
  DataType type_;
  AssignExpression* ae_;
};

} //namespace RTC
} //namespace midend

#endif
