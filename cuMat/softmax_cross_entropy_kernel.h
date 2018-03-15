/*!
  @file softmax_cross_entropy_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef SOFTMAX_CROSS_ENTROPY_KERNEL_H_
#define SOFTMAX_CROSS_ENTROPY_KERNEL_H_

__global__ void softmax_cross_entropy_kernel(const float * __restrict__ src1,
                                              const float * __restrict__ src2,
                                              float * __restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    void softmax_cross_entropy_kernel_exec(const float *src1, const float *src2, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
