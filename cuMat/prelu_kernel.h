#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef PRELU_KERNEL_H_
#define PRELU_KERNEL_H_

#ifdef __cplusplus
extern "C"{
#endif
    void prelu_kernel_exec(const float *src, const float *a, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
