#include <cuda_runtime.h>
#include <cstdlib>

#include "add.h"

__global__ void MatAdd(float **A, float **B, float **C){
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

void MatAdd_exec(float **lvalue1, float **lvalue2, float **rvalue){
    float **dev_l1, **dev_l2, **dev_r;

    cudaMalloc((void**)&dev_l1, N*N*sizeof(float));
    cudaMalloc((void**)&dev_l2, N*N*sizeof(float));
    cudaMalloc((void**)&dev_r,  N*N*sizeof(float));

    cudaMemcpy(dev_l1, lvalue1, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_l2, lvalue2, N*N*sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);

    MatAdd<<<numBlocks, threadsPerBlock>>>(dev_l1, dev_l2, dev_r);

    cudaMemcpy(rvalue, dev_r, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_l1);
    cudaFree(dev_l2);
    cudaFree(dev_r);
}
