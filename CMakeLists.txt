project(DeepLearning)

cmake_minimum_required(VERSION 2.8.3)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  )

add_subdirectory(lib/cuMat)

link_directories(
  ${CUDA_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  )

# executable
add_definitions(-std=c++11)
set(LINK_LIBRARIES
  ${CUDA_LIBRARAIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${Boost_LIBRARIES}
  )

add_executable(${PROJECT_NAME}_test1 src/test1.cpp)
add_executable(${PROJECT_NAME}_test2 src/test2.cpp)

target_link_libraries(${PROJECT_NAME}_test1
  ${LINK_LIBRARIES}
  cuMat
  )

target_link_libraries(${PROJECT_NAME}_test2
  ${LINK_LIBRARIES}
  cuMat
  )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
