#ifndef CAVS_MIDEND_DEVICES_H_
#define CAVS_MIDEND_DEVICES_H_

#include "cavs/proto/types.pb.h"
#include "cavs/midend/tensor.h"

namespace midend {

const char* DeviceTypeToString(DeviceType type);

class Tensor;
class DeviceContext {
 public:
  static void MemcpyHostToDevice(Tensor* out, const Tensor& inp);
  static void MemcpyDeviceToHost(Tensor* out, const Tensor& inp);
};

} //namespace midend 

#endif
