#include "mat_mod_kernel.h"
#include "def.h"

static const int block_size = 32;

__global__ void mat_mod_kernel(const float * __restrict__ src,
                               float * __restrict__ dst, int m, int n, float p){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n){
        dst[row*n+col] = p / (src[row*n+col]+eps);
    }
}

void mat_mod_kernel_exec(const float *src, float *dst, int m, int n, float p){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x - 1)/block.x, (m + block.y - 1)/block.y);

    mat_mod_kernel <<<grid, block>>>(src, dst, m, n, p);
    cudaThreadSynchronize();
}
