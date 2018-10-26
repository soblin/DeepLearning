/*!
  @file relu_d_kernel.h
 */
#include "cuda_runtime.h"

#ifndef RELU_D_KERNEL_H_
#define RELU_D_KERNEL_H_

__device__ __forceinline__ float relu_d(float a);

__global__ void relu_d_kernel(const float * __restrict__ src,
                              float * __restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates \f$ \mathrm{dst[i][j] = (src[i][j] > 1.0)? 1.0 : 0.0} \f$
     */
    void relu_d_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
