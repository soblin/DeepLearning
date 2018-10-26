#include "prelu_d_kernel.h"

static const int block_size = 32;

__global__ void prelu_d_kernel(const float *__restrict__ src,
                               const float *__restrict__ a,
                               float *__restrict__ dst, float *__restrict__ da, int m, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n){
        int idx = row * n + col;
        dst[idx] = src[idx] > 0.0f ? 1.0f : a[idx];
        da[idx] = src[idx] > 0.0f ? 0.0f : src[idx];
    }
}

void prelu_d_kernel_exec(const float *src, const float *a, float *dst , float *da, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (m + block.y-1)/block.y);

    prelu_d_kernel <<< grid, block >>> (src, a, dst, da, m, n);
    cudaThreadSynchronize();
}
