/*!
  @file prelu_d_kernel.h
 */
#include "/usr/local/cuda-9.1/include/cuda_runtime.h"

#ifndef PRELU_D_KERNEL_H_
#define PRELU_D_KERNEL_H_

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates {dst[i][j], da[i][j]} = (src[i][j] > 0.0)? {1.0, 0.0} : {a[i][j], src[i][j]}
     */
    void prelu_d_kernel_exec(const float *src, const float *a, float *dst, float *da, int m ,int n);
#ifdef __cplusplus
};
#endif

#endif
