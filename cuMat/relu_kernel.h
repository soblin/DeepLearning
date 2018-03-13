/*!
  @file relu_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef RELU_KERNEL_H_
#define RELU_KERNEL_H_

__device__ __forceinline__ float relu(float a);

__global__ void relu_kernel(const float * __restrict__ src,
                            float * __restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates dst[i][j] = (src[i][j] > 1.0)? dst[i][j] : 0.0
     */
    void relu_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
