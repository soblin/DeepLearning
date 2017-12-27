#include <cuda_runtime.h>

#ifndef ADD_H_
#define ADD_H_

const int N = 10;

__global__ void MatAdd(float **A, float **B, float **C);

#ifdef __cplusplus
extern "C"{
#endif
    void MatAdd_exec(float **lvalue1, float **lvalue2, float **rvalue);
#ifdef __cplusplus
};
#endif


#endif
