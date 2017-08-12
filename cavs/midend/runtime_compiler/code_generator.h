#ifndef CAVS_MIDEND_RUNTIME_COMPILER_CODE_GENERATOR_H_
#define CAVS_MIDEND_RUNTIME_COMPILER_CODE_GENERATOR_H_

#include "cavs/midend/runtime_compiler/parser.h"

namespace midend {
namespace RTC {

class CodeGenerator {
 public:
  CodeGenerator(std::list<Node*>* n, std::vector<std::vector<int>>* dependency);
  inline static std::string PrefixedVar(std::string var) {
    return "tmp_" + var; 
  }
  inline static std::string arrSize(std::string arr) {
    return arr + "_count";
  }
  inline static std::string typeToString(DataType type) {
    CHECK(DataTypeToString.find((int)type) != DataTypeToString.end());
    return DataTypeToString.at((int)type);
  }
  
 private:
  std::vector<std::string> kernel_source_;
  Parser parser_;
  static std::unordered_map<int, std::string> DataTypeToString;
};

} //namespace RTC
} //namespace midend

#endif

