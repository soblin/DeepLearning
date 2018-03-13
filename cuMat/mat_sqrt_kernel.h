/*!
  @file mat_sqrt_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef MAT_SQRT_KERNEL_H_
#define MAT_SQRT_KERNEL_H_

__device__ __forceinline__ float mat_sqrt(float a, float alpha);

__global__ void mat_sqrt_kernel(const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, float alpha);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief \f$ \mathrm{dst[i][j] = \sqrt{src[i][j]+\alpha}} \f$
     */
    void mat_sqrt_kernel_exec(const float *src, float *dst, int m, int n, float alpha);
#ifdef __cplusplus
};
#endif

#endif
