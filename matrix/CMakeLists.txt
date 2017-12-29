cmake_minimum_required(VERSION 3.3.2)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-Wall -g")

find_package(Eigen3 REQUIRED)

project(a.out)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(EIGEN3_INCLUDE_DIR)

add_executable(a.out eigen-test.cc)
