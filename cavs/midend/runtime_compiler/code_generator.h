#ifndef CAVS_MIDEND_RUNTIME_COMPILER_CODE_GENERATOR_H_
#define CAVS_MIDEND_RUNTIME_COMPILER_CODE_GENERATOR_H_

#include "cavs/midend/runtime_compiler/parser.h"

namespace midend {
namespace RTC {

class CodeGenerator {
 public:
  CodeGenerator(std::list<Node*>* n);
  static std::string PrefixedVar(std::string var) {
    return "tmp_" + var; 
  }
  
 private:
  std::vector<std::string> kernel_source_;
  Parser parser_;
};

} //namespace RTC
} //namespace midend

#endif

