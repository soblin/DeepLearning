/*!
  @file mat_inverse_d_kernel.h
 */
#include "cuda_runtime.h"

#ifndef MAT_INVERSE_D_KERNEL_H_
#define MAT_INVERSE_D_KERNEL_H_

__device__ __forceinline__ float mat_inverse_d(float a);

__global__ void mat_inverse_d_kernel(const float *__restrict__ src,
                                     float *__restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief \f$ \mathrm{dst[i][j] = -1.0/src[i][j]^{2}} \f$
     */
    void mat_inverse_d_kernel_exec(const float *src, float *dst, int m ,int n);
#ifdef __cplusplus
};
#endif

#endif
