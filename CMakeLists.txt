cmake_minimum_required(VERSION 3.10)

project(op-kernel-cc-rm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE headers ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE cuda_headers ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_executable(
    ${PROJECT_NAME}
    # ${sources}
    # ${cuda_sources}
    # main.cu
    main.cpp
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        cuda 
)

# Specify options for CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
