#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef MAT_COS_KERNEL_H_
#define MAT_COS_KERNEL_H_

__device__ __forceinline__ float mat_cos(float a, float alpha);

__global__ void mat_cos_kernel(const float * __restrict__ src,
                               float * __restrict__ dst, int m, int n, float alpha);

#ifdef __cplusplus
extern "C"{
#endif
    void mat_cos_kernel_exec(const float *src, float *dst, int m, int n, float alpha);
#ifdef __cplusplus
};
#endif

#endif
