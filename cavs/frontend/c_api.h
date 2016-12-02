#ifndef CAVS_FRONTEND_C_API_H_
#define CAVS_FRONTEND_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif

//#include "cavs/midend/session.h"

typedef enum {
  F_FLOAT = 1,
  F_DOUBLE = 2,
  F_INT32 = 3,  
} F_Dtype;

typedef struct F_Session F_Session;

extern F_Session* F_NewSession(const string& name);
extern void F_SetOpChainOp(F_Session* s, 
                           const void* proto, size_t len);
extern void F_Run(F_Session* s);

#ifdef __cplusplus
} //end extern "C"
#endif

#endif
