cmake_minimum_required(VERSION 3.3.2)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

find_package(Boost REQUIRED)

include_directories(${EIGEN3_INLCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

project(a.out)

add_executable(a.out function.cc)
