#ifndef DEVICES_H
#define DEVICES_H

#include "cavs/midend/types.pb.h"
#include "cavs/midend/tensor.h"

namespace cavs {

const char* DeviceTypeToString(DeviceType type);

class DeviceContext {

};

class CPUContext : public DeviceContext {

};

class GPUContext : public DeviceContext {
 public:
  void MemcpyHostToDevice(Tensor* out, const Tensor& inp);
  void MemcpyDeviceToHost(Tensor* out, const Tensor& inp);
};

} //namespace cavs

#endif
