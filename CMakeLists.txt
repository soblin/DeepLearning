cmake_minimum_required(VERSION 2.8.3)

# library
add_subdirectory(lib)

find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  )

link_directories(
  ${CUDA_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  )

# executable
add_definitions(-std=c++11)
set(LINK_LIBRARIES
  ${CUDA_LIBRARAIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${Boost_LIBRARIES}
  )

add_executable(main src/main.cpp)

target_link_libraries(main
  ${LINK_LIBRARIES}
  cuMat
  )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
