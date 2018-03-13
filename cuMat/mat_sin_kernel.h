/*!
  @file mat_sin_kernel.h
*/
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef MAT_SIN_KERNEL_H_
#define MAT_SIN_KERNEL_H_

__device__ __forceinline__ float mat_sin(float a, float alpha);

__global__ void mat_sin_kernel(const float * __restrict__ src,
                               float * __restrict__ dst, int m, int n, float alpha);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief \f$ \mathrm{dst[i][j] = \sin(src[i][j])} \f$
     */
    void mat_sin_kernel_exec(const float *src, float *dst, int m, int n, float alpha);
#ifdef __cplusplus
};
#endif

#endif
