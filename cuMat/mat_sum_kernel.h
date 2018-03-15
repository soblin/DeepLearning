/*!
  @file mat_sum_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef MAT_SUM_KERNEL_H_
#define MAT_SUM_KERNEL_H_

__global__ void mat_sun_kernel(const float *__restrict__ src,
                               float *__restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    void mat_sum_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
