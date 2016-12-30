#include "cavs/midend/devices.h"
#include "cavs/util/macros_gpu.h"

namespace midend {

void DeviceContext::MemcpyHostToDevice(Tensor* out, const Tensor& inp) {
  checkCudaError(cudaMemcpy(out->buf_->data(), inp.buf_->data(), 
                 inp.buf_->size(), 
                 cudaMemcpyHostToDevice));
}

void DeviceContext::MemcpyDeviceToHost(Tensor* out, const Tensor& inp) {
  checkCudaError(cudaMemcpy(out->buf_->data(), inp.buf_->data(), 
                 inp.buf_->size(), 
                 cudaMemcpyDeviceToHost));

}

} //namespace midend 
