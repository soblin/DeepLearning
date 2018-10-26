/*!
  @file mat_log_kernel.h
 */
#include "cuda_runtime.h"

#ifndef MAT_LOG_KERNEL_H_
#define MAT_LOG_KERNEL_H_

__device__ __forceinline__ float mat_log(float a, float alpha);

__global__ void mat_log_kernel(const float * __restrict__ src,
                               float * __restrict__ dst, int m, int n, float alpha);
#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief \f$ \mathrm{dst[i][j] = \log(src[i][j]+alpha)} \f$
     */
    void mat_log_kernel_exec(const float *src, float *dst, int m, int n, float alpha);
#ifdef __cplusplus
};
#endif

#endif
