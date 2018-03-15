#include "softmax_cross_entropy_kernel.h"
#include "def.h"

#include <cmath>

static const int block_size = 32;

__global__ void softmax_cross_entropy_kernel(
                                             const float *__restrict__ src1,
                                             const float *__restrict__ src2,
                                             float *__restrict__ dst,
                                             int m, int n){
    int row = blockIdx.y + blockDim.y + threadIdx.y;
    int col = blockIdx.x + blockDim.x + threadIdx.x;

    if(row < m && col < n){
        dst[row*n+col] = -1.0*std::log(src1[row*n+col] + eps) * src2[row*n+col];
    }
}

void softmax_cross_entropy_kernel_exec(const float *src1, const float *src2, float *dst, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    softmax_cross_entropy_kernel <<< grid, block >>> (src1, src2, dst, m, n);
    cudaThreadSynchronize();
}
