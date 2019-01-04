#include "DeepLearning/cuMat/mat_inverse_d_kernel.h"

static const int block_size = 32;

__device__ __forceinline__ float mat_inverse_d(float a){
    return -1.0 / (a+1e-8)*(a+1e-8);
}

__global__ void mat_inverse_d_kernel(const float *__restrict__ src,
                                     float *__restrict__ dst, int m, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n){
        dst[row*n+col] = mat_inverse_d(src[row*n+col]);
    }
}

void mat_inverse_d_kernel_exec(const float *src, float *dst, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    mat_inverse_d_kernel <<< grid, block >>> (src, dst, m, n);
    cudaThreadSynchronize();
}
