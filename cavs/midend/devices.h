#ifndef DEVICES_H
#define DEVICES_H

#include "cavs/midend/types.pb.h"
#include "cavs/midend/tensor.h"

namespace cavs {

const char* DeviceTypeToString(DeviceType type);

class Tensor;
class DeviceContext {
 public:
  static void MemcpyHostToDevice(Tensor* out, const Tensor& inp);
  static void MemcpyDeviceToHost(Tensor* out, const Tensor& inp);
};

} //namespace cavs

#endif
