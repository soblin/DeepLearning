#include "mat_sqrt_kernel.h"

#include <cmath>

static const int block_size = 32;

__device__ __forceinline__ float mat_sqrt(float a, float alpha){
    return std::sqrt(a + alpha);
}

__global__ void mat_sqrt_kernel(const float *__restrict__ src,
                                float * __restrict__ dst, int m, int n, float alpha){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n){
        dst[row * n + col] = mat_sqrt(src[row * n + col], alpha);
    }
}

void mat_sqrt_kernel_exec(const float *src, float *dst, int m, int n, float alpha){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (m + block.y-1)/block.y);

    mat_sqrt_kernel <<< grid, block >>> (src, dst,m, n, alpha);
    cudaThreadSynchronize();
}
