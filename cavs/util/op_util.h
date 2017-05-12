#include "cavs/proto/op_def.pb.h"

#include <vector>
#include <string>

template<typename T>
T GetSingleArg(const OpDef& op_def, const std::string& key); 

template<typename T>
T GetSingleArg(const OpDef& op_def, const std::string& key, T value); 

template<typename T>
std::vector<T> GetListArg(const OpDef& op_def, const std::string& key); 

std::string GetGradientName(const std::string& op);

std::string GetOriginName(const std::string& op);

size_t GetHash(const OpDef& op_def);

bool IsVariable(const std::string& edge);
