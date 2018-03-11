/*!
  @file mat_inverse_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef MAT_INVERSE_KERNEL_H_
#define MAT_INVERSE_KERNEL_H_

__device__ __forceinline__ float mat_inverse(float a);

__global__ void mat_inverse_kernel(const float *__restrict__ src,
                                  float *__restrict__ dst, int m, int n);

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates dst[i][j] = 1.0/(src[i][j]+1e-8)
     */
    void mat_inverse_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus   
};
#endif

#endif
