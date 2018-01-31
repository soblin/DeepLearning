#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef SOFTMAX_KERNEL_H_
#define SOFTMAX_KERNEL_H_

__device__ __forceinline__ float softmax(float a, float sum);

__global__ void softmax_kernel(const float *__restrict__ src,
                               float *__restrict__ dst, int m, int n, float *sum, float *max);

__global__ void softmx_kernel2(const float *__restrict__ src,
                               float *__restrict__ dst, int m, int n, float *sum, float *max);

__global__ void softmax_kernel3(const float *__restrict__ src, int m, int n, float *sum, float *max);

#ifdef __cplusplus
extern "C"{
#endif
    void softmax_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
