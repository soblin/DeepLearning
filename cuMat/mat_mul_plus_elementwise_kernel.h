#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef MAT_MUL_PLUS_ELEMENTWISE_KERNEL_H_
#define MAT_MUL_PLUS_ELEMENTWISE_KERNEL_H_

__global__ void mat_mul_plus_elementwise_kernel(const float * __restrict__ src1, const float * __restrict__ src2, float * __restrict__ dst, float alpha, float bet, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    void mat_mul_plus_elementwise_kernel_exec(const float *src1, const float *src2, float *dst, float alpha, float beta, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
