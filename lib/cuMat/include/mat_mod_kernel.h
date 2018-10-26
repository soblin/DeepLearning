/*!
  @file mat_mod_kernel.h
 */
#include "cuda_runtime.h"

#ifndef MAT_MOD_KERNEL_H_
#define MAT_MOD_KERNEL_H_

__global__ void mat_mod_kernel(const float * __restrict__ src,
                               float * __restrict__ dst, int m, int n, float p);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief \f$ \mathrm{dst[i][j] = p / src[i][j]} \f$
     */
    void mat_mod_kernel_exec(const float *src, float *dst, int m, int n, float p);
#ifdef __cplusplus   
};
#endif

#endif
