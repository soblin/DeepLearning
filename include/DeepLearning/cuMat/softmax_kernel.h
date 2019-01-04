/*!
  @file softmax_kernel.h
*/
#include "cuda_runtime.h"

#ifndef SOFTMAX_KERNEL_H_
#define SOFTMAX_KERNEL_H_

__device__ void AtomicMax(float *const address, const float value);
/*!
  @brief returns \f$ \mathrm{a / (sum + 1e-8)} \f$
*/
__device__ __forceinline__ float softmax(float a, float sum);

__global__ void softmax_kernel(const float *__restrict__ src,
                               float *__restrict__ dst, int m, int n, float *sum, float *max);

/*!
  @brief This function calculates \f$ \mathrm{dst[i][j] = exp(src[i][j] - max[i])} \f$.
  At the same time, the array sum stores the summation \f$ \mathrm{sum[i] = exp(src[i][0]-max[i]) + \cdots + exp(src[i][n-1]-max[i])} \f$
*/
__global__ void softmx_kernel2(const float *__restrict__ src,
                               float *__restrict__ dst, int m, int n, float *sum, float *max);

/*!
  @brief This function searches for the max value for each row of src and stores the maximum value to array max.
  So max[i] contains the maximum value of i-th row of src.
*/
__global__ void softmax_kernel3(const float *__restrict__ src, int m, int n, float *sum, float *max);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief Softmax function takes an array as its arguments returns the same size array. We define the softmax function for matrix by recognizing each rows as the arguments.
      \f$ \mathrm{dst[i][j] = softmax(src[i])[j]} \f$
    */
    void softmax_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
