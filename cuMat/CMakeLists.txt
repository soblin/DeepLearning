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
  )
