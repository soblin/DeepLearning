#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef MAT_SQRT_D_KERNEL_H_
#define MAT_SQRT_D_KERNEL_H_

__device__ __forceinline__ float mat_sqrt_d(float a, float alpha);

__global__ void mat_sqrt_d_kernel(const float * __restrict__ src,
                                  float * __restrict__ dst, int m, int n, float alpha);

#ifdef __cplusplus
extern "C"{
#endif
    void mat_sqrt_d_kernel_exec(const float *src, float *dst, int m, int n, float alpha);
#ifdef __cplusplus
};
#endif

#endif
