#include "prelu_kernel.h"

static const int block_size = 32;

__device__ __forceinline__ float prelu(float x, float a){
    return x > 0.0f ? x : a*x;
}

__global__ void prelu_kernel(const float * __restrict__ src,
                             const float * __restrict__ a,
                             float * __restrict__ dst, int m, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n){
        int idx = row * n + col;
        dst[idx] = prelu(src[idx], a[idx]);
    }
}

void prelu_kernel_exec(const float *src, const float *a, float *dst, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (m + block.y-1)/block.y);

    prelu_kernel <<< grid, block >>> (src, a, dst, m, n);
    cudaThreadSynchronize();
}
