cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cs521_mp1 LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 89)

find_package(CUDAToolkit REQUIRED)
add_executable(mp1_gpu gemm_gpu.cu)
target_link_libraries(mp1_gpu PRIVATE CUDA::cublas)