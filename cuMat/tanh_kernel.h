/*!
  @file tanh_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef TANH_KERNEL_H_
#define TANH_KERNEL_H_

__device__ __forceinline__ float tanh_f(float a);

__global__ void tanh_kernel(const float *__restrict__ src,
                            float *__restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates \f$dst[i][j] = \tanh(src[i][j])\f$
     */
    void tanh_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
