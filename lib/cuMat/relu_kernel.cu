#include "DeepLearning/cuMat/relu_kernel.h"

static const int block_size = 32;

__device__ __forceinline__ float relu(float a){
    return a > 0.0f? a : .0f;
}

__global__ void relu_kernel(const float * __restrict__ src,
                            float * __restrict__ dst, int m, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n) dst[row * n + col] = relu(src[row * n + col]);
}

void relu_kernel_exec(const float *src, float *dst, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (m + block.y-1)/block.y);

    relu_kernel <<< grid, block >>> (src, dst, m, n);
    cudaThreadSynchronize();
}
