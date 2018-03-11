/*!
  @file mat_exp_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef MAT_EXP_KERNEL_H_
#define MAT_EXP_KERNEL_H_

__device__ __forceinline__ float mat_exp(float a, float alpha);

__global__ void mat_exp_kernel(const float *__restrict__ src,
                                               float * __restrict__ dst, int m, int n, float alpha);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates dst[i][j] = exp(src[i][j]+alpha)
     */
    void mat_exp_kernel_exec(const float *src, float *dst, int m, int n, float alpha);
#ifdef __cplusplus
};
#endif

#endif 
