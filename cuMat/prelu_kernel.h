/*!
  @file prelu_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef PRELU_KERNEL_H_
#define PRELU_KERNEL_H_

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates dst[i][j] = (src[i][j] > 0.0)? src[i][j] : src[i][j]*a[i][j]
     */
    void prelu_kernel_exec(const float *src, const float *a, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
