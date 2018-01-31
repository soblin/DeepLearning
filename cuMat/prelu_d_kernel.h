#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef PRELU_D_KERNEL_H_
#define PRELU_D_KERNEL_H_

#ifdef __cplusplus
extern "C"{
#endif
    void prelu_d_kernel_exec(const float *src, const float *dst, float *dst, float *da, int m ,int n);
#ifdef __cplusplus
};
#endif

#endif
