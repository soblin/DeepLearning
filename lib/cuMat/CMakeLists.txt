cmake_minimum_required(VERSION 2.8.3)

project(cuMat)

find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE cuMat_SRCS "src/*.cu")
file(GLOB_RECURSE cuMat_HDRS "include/*.h")

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

link_directories(
  ${CUDA_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CUDA_ARCH -gencode arch=compute_62,code=sm_62)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -Xcompiler -fPIC)

CUDA_ADD_LIBRARY(
  cuMat
  SHARED
  ${cuMat_SRCS}
  )

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
