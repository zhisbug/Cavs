#ifndef CAVS_FRONTEND_C_API_H_
#define CAVS_FRONTEND_C_API_H_

#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  C_FLOAT = 0,
  C_DOUBLE = 1,
  C_INT32 = 2,  
} C_Dtype;

typedef struct C_Session  C_Session;
typedef struct C_Tensor   C_Tensor;
typedef struct C_Scope    C_Scope;

extern C_Session* C_NewSession(
    const char* name, size_t name_len, int opt);
extern C_Tensor* C_NewTensor(const char* name, size_t name_len, 
    const int* shape, int dims, C_Dtype dtype);
//extern void C_DumpGraph(C_DepGraph* c_graph);
extern void C_AddOp(const void* def, size_t def_length,
    int** dim, size_t* dim_length);
extern void C_AddOptimizerOp(
    const void* def, size_t def_length);
extern void C_AddFunction(const void* def, size_t def_length,
    int** dim, size_t* dim_length);
extern void C_AddControlDependency(const void* def, size_t def_length);
extern void C_Run(C_Session* s, 
    const char** c_output_names, C_Tensor** c_output_tensors, int noutputs,
    const char** c_input_names, C_Tensor* const* c_input_tensors, int ninputs);
extern void* C_TensorData(const C_Tensor* t);
extern size_t C_TensorSize(const C_Tensor* t);

#ifdef __cplusplus
} //end extern "C"
#endif

#endif
