#include "DeepLearning/cuMat/mat_mul_plus_elementwise_kernel.h"

static const int block_size = 32;

__global__ void mat_mul_plus_elementwise_kernel(const float * __restrict__ src1, const float * __restrict__ src2, float * __restrict__ dst, float alpha, float beta, int m, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n){
	dst[row*n + col] += alpha*src1[row*n+col]*beta*src2[row*n+col];
    }
}

void mat_mul_plus_elementwise_kernel_exec(const float *src1, const float *src2, float *dst, float alpha, float beta, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (m + block.y-1)/block.y);

    mat_mul_plus_elementwise_kernel<<<grid, block>>>(src1, src2, dst, alpha, beta, m, n);
    cudaThreadSynchronize();
}
