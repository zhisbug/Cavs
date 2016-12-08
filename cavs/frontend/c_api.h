#ifndef CAVS_FRONTEND_C_API_H_
#define CAVS_FRONTEND_C_API_H_

#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

//#include "cavs/midend/session.h"

typedef enum {
  F_FLOAT = 0,
  F_DOUBLE = 1,
  F_INT32 = 2,  
} F_Dtype;

typedef struct F_Session F_Session;
typedef struct F_Tensor F_Tensor;

extern F_Session* F_NewSession(const char* name, size_t name_len, 
    const void* proto, size_t proto_len);
extern F_Tensor* F_NewTensor(const char* name, size_t name_len, 
    const int* shape, int dims, F_Dtype dtype);
//extern void F_SetOpChainOp(F_Session* s, 
      //const void* proto, size_t len);
extern void F_Run(F_Session* s, 
      const char** c_output_names, F_Tensor** c_output_tensors, int noutputs,
      const char** c_input_names, F_Tensor* const* c_input_tensors, int ninputs);
extern void* F_TensorData(const F_Tensor* t);
extern size_t F_TensorSize(const F_Tensor* t);
//extern F_Tensor* F_GetTensorFromSession(
      //F_Session* sess, const char* c_tensor_name, size_t len); 

#ifdef __cplusplus
} //end extern "C"
#endif

#endif
