#include "DeepLearning/cuMat/mat_l2_kernel.h"

static const int block_size = 32;

__global__ void mat_l2_kernel(const float *__restrict__ src,
                              float *__restrict__ dst, int m, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n) atomicAdd(dst, src[row * n + col]*src[row * n + col]);
}

void mat_l2_kernel_exec(const float *src, float *dst, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (m + block.y - 1)/block.y);

    mat_l2_kernel <<< grid, block >>> (src, dst, m, n);
    cudaThreadSynchronize();
}
