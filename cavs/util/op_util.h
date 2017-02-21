#include "cavs/proto/op_def.pb.h"

#include <vector>
#include <string>

template<typename T>
T GetSingleArg(const OpDef& op_def, const std::string& key); 

template<typename T>
T GetSingleArg(const OpDef& op_def, const std::string& key, T value); 

template<typename T>
std::vector<T> GetListArg(const OpDef& op_def, const std::string& key); 
