/*!
  @file tanh_d_kernel.h
 */
#include "cuda_runtime.h"
#ifndef TANH_D_KERNEL_H_
#define TANH_D_KERNEL_H_

__device__ __forceinline__ float tanh_d(float a);

__global__ void tanh_d_kernel(const float * __restrict__ src,
                              float *__restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates \f$ \mathrm{dst[i][j] = 1.0 - tanh(src[i][j])*tanh(src[i][j])} \f$
     */
    void tanh_d_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
