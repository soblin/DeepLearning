/*!
  @file prelu_kernel.h
 */
#include "cuda_runtime.h"

#ifndef PRELU_KERNEL_H_
#define PRELU_KERNEL_H_

#ifdef __cplusplus
extern "C"{
#endif
    /*!
      @brief This operates \f$ \mathrm{dst[i][j] = (src[i][j] > 0.0)? src[i][j] : src[i][j]*a[i][j]} \f$
     */
    void prelu_kernel_exec(const float *src, const float *a, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif
