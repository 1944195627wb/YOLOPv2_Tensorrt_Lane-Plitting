cmake_minimum_required(VERSION 3.10)
PROJECT(Linfer VERSION 1.0.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(CMAKE_SKIP_BUILD_RPATH False)
set(CMAKE_SKIP_RPATH False)
set(CMAKE_BUILD_RPATH "/usr/lib/aarch64-linux-gnu")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")
find_package(CUDA 11.4 REQUIRED)

file(GLOB_RECURSE CPPS
        ${PROJECT_SOURCE_DIR}/apps/*.cpp
        ${PROJECT_SOURCE_DIR}/apps/*.cu
        ${PROJECT_SOURCE_DIR}/trt_common/*.cpp
        ${PROJECT_SOURCE_DIR}/trt_common/*.cu
        )

set(OpenCV_DIR "/usr/include/opencv4/opencv2")
set(CUDA_DIR "/usr/local/cuda-11.4")
set(TENSORRT_DIR "/usr/src/tensorrt")
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_DIR}/include
        ${TENSORRT_DIR}/include
        "/usr/include/eigen3"
)
link_directories(
        ${CUDA_DIR}/lib64
        ${TENSORRT_DIR}/lib
)
list(APPEND ALL_LIBS
        cuda cublas cudart cudnn
        nvinfer nvinfer_plugin nvonnxparser
        pthread ${OpenCV_LIBS}
        )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wfatal-errors -pthread -w")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -g -O0 -Xcompiler -fPIC")

add_library(${PROJECT_NAME} SHARED ${CPPS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})

# reference：https://developer.nvidia.com/cuda-gpus#compute
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 89)
target_compile_options(${PROJECT_NAME} PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -lineinfo --use_fast_math --disable-warnings>)

add_executable(pro main.cpp)
target_link_libraries(pro ${PROJECT_NAME} ${ALL_LIBS})
