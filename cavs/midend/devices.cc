#include "cavs/midend/devices.h"

const char* DeviceTypeToString(DeviceType type) {
  if (type == GPU)
    return "GPU";
  else if (type == CPU)
    return "CPU";
}
