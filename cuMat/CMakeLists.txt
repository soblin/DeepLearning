cmake_minimum_required(VERSION 3.3.2)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-Wall -g")

project(cumat)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${CUDA_LIBRARY_DIRS})

set(CUDA_ARCH -gencode arch=compute_50,code=sm_50)

#なぜかこの環境ではこれが必要
set(CUDA_64_BIT_DEVICE_CODE ON)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH}  -Xcompiler -fPIC)

CUDA_ADD_LIBRARY(
  cumat
  SHARED
  mat_ones_kernel.cu
  mat_mul_elementwise_kernel.cu
  mat_mul_plus_elementwise_kernel.cu
  slice_rows_kernel.cu
  mat_log_kernel.cu
  mat_sqrt_kernel.cu
  mat_sqrt_d_kernel.cu
  mat_cos_kernel.cu
  mat_sin_kernel.cu
  relu_kernel.cu
  relu_d_kernel.cu
  prelu_kernel.cu
  sigmoid_kernel.cu
  sigmoid_d_kernel.cu
  softmax_kernel.cu
  tanh_kernel.cu
  tanh_d_kernel.cu
  mat_l2_kernel.cu
  )
