/*!
  @file mat_div_kernel.h
 */
#include "cuda_runtime.h"

#ifndef MAT_DIV_KERNEL_H_
#define MAT_DIV_KERNEL_H_

__global__ void mat_div_kernel(const float * __restrict__ src1,
                               const float * __restrict__ src2,
                               float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief \f$ \mathrm{dst[i][j] = src1[i][j] / src2[i][j]} \f$
     */
    void mat_div_kernel_exec(const float *src1, const float *src2, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
