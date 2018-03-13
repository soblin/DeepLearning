/*!
  @file sigmoid_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef SIGMOID_KERNEL_H_
#define SIGMOID_KERNEL_H_

__device__ __forceinline__ float sigmoid(float a);

__global__ void sigmoid_kernel(const float *__restrict__ src,
                               float *__restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates dst[i][j] = 1.0 / (1.0 + exp(-src[i][j]))
     */
    void sigmoid_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
