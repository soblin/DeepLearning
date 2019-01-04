#include "DeepLearning/cuMat/slice_rows_kernel.h"
#include <stdio.h>
#define BLOCK_SIZE 32

__global__ void slice_rows_kernel(const float *__restrict__ src,
                                  float *__restrict__ dst, int m, int n, int offset, int len){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(offset <= row && row < offset + len && row < n && col < m){
        dst[col * len + row - offset] = src[col * n + row];
    }
}

void slice_rows_kernel_exec(const float *src, float *dst ,int m, int n, int offset, int len){
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m + block.x-1)/block.x, (n+block.y-1)/block.y);

    slice_rows_kernel <<< grid, block >>> (src, dst, m, n, offset, len);
    cudaThreadSynchronize();
}

__global__ void join_rows_kernel(const float *__restrict__ src,
                                 float * __restrict__ dst, int m, int n, int offset, int len){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(offset <= row && row < offset + len && row < n && col < m){
        dst[col * n + row] = src[col * len + row - offset];
    }
}

void join_rows_kernel_exec(const float *src, float *dst, int m, int n, int offset, int len){
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m + block.x-1)/block.x, (n + block.y-1)/block.y);

    join_rows_kernel <<< grid, block >>> (src, dst, m, n, offset, len);
    cudaThreadSynchronize();
}
