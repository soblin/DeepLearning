#include "DeepLearning/cuMat/mat_ones_kernel.h"

static const int block_size = 32;

__global__ void mat_ones_kernel(float * __restrict__ src,
				float * __restrict__ dst,
				const int m, const int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n) dst[row * n + col] = 1.0;
}

void mat_ones_kernel_exec(float *src, float *dst, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x - 1)/block.x, (m + block.y-1)/block.y);

    mat_ones_kernel<<< grid, block >>>(src, dst, m, n);
    cudaThreadSynchronize();
}
