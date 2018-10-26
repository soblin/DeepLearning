/*!
  @file mat_ones_kernel.h
 */
#include "cuda_runtime.h"

#ifndef MAT_ONES_KERNEL_H_
#define MAT_ONES_KERNEL_H_

__global__ void mat_ones_kernel(float * __restrict__ src,
				float * __restrict__ dst,
				const int m, const int n);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief \f$ \mathrm{dst[i]][j] = 1.0} \f$
     */
    void mat_ones_kernel_exec(float *src, float *dst, const int m, const int n);
#ifdef __cplusplus    
};
#endif

#endif
