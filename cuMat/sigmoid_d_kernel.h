#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef SIGMOID_D_KERNEL_H_
#define SIGMOID_D_KERNEL_H_

__device__ __forceinline__ float sigmoid_d(float a);

__global__ void sigmoid_d_kernel(const float *__restrict__ src,
                                float *__restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    void sigmoid_d_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
