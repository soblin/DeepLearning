#include "DeepLearning/cuMat/softmax_kernel.h"

#include <cmath>

static const int block_size = 32;

__device__ void AtomicMax(float *const address, const float value){
    if(*address >= value) return;
    int *const address_as_i = (int*)address;
    int old = *address_as_i,assumed;

    do{
        assumed = old;
        if(__int_as_float(assumed) >= value) break;
        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
    }while(assumed != old);
}

__device__ __forceinline__ float softmax(float a, float sum){
    return a / (sum + 1e-8);
}

__global__ void softmax_kernel(const float *__restrict__ src,
                               float *__restrict__ dst, int m, int n, float *sum, float *max){
    int row = blockIdx.y + blockDim.y + threadIdx.y;
    int col = blockIdx.x + blockDim.x + threadIdx.x;

    if(row < m && col < n) dst[row * n + col] = softmax(dst[row * n + col], sum[row]);
}

__global__ void softmax_kernel2(const float *__restrict__ src,
                                float *__restrict__ dst, int m, int n, float *sum, float *max){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n){
        float a = std::exp(src[row * n + col] - max[row]);
        atomicAdd(&sum[row], a);
        dst[row * n + col] = a;
    }
}

__global__ void softmax_kernel3(const float *__restrict__ src, int m, int n, float *max){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n) AtomicMax(&max[row], src[row * n + col]);
}

void softmax_kernel_exec(const float *src, float *dst, int m, int n){
    dim3 block(block_size, block_size);
    dim3 grid((n + block.x-1)/block.x, (m + block.y-1)/block.y);

    float *max, *sum;

    cudaError_t error = cudaMalloc((void**)&max, m * sizeof(*max));
    error = cudaMalloc((void**)&sum, m * sizeof(*max));
    cudaThreadSynchronize();
    cudaMemset(max, 0x00, m * sizeof(*max));
    cudaMemset(sum, 0x00, m * sizeof(*sum));

    softmax_kernel3 <<< grid, block >>> (src, m, n, max);
    cudaThreadSynchronize();
    softmax_kernel2 <<< grid, block >>> (src, dst, m, n ,sum, max);
    cudaThreadSynchronize();
    softmax_kernel <<< grid, block >>> (src, dst, m, n, sum, max);
    cudaThreadSynchronize();
    cudaFree(max);
    cudaFree(sum);
}
