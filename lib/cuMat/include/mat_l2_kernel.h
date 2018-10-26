/*!
  @file mat_l2_kernel.h
 */
#include "cuda_runtime.h"

#ifndef MAT_L2_KERNEL_H_
#define MAT_L2_KERNEL_H_

__global__ void mat_l2_kernel(const float *__restrict__ src,
                              float *__restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This returns the L2-norm of matrix.
     */
    void mat_l2_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
