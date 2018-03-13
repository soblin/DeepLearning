#include "mat_div_kernel.h"

static const int block_size = 32;
//apply dst[i][j] = src1[i][j] / src2[i][j]
__global__ void mat_div_kernel(const float * __restrict__ src1,
                               const float * __restrict__ src2,
                               float * __restrict__ dst, int m, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n)
        dst[row * n + col] = src1[row * n + col] / src2[row * n + col];
}

void mat_div_kernel_exec(const float *src1, const float *src2, float *dst, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (m + block.y -1)/block.y);

    mat_div_kernel <<< grid, block >>> (src1, src2, dst, m, n);
    cudaThreadSynchronize();
}
