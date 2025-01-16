cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MGG LANGUAGES CXX CUDA)


#############################################################################

set($ENV{PATH} "/home/linzhiheng/thirdparty/openmpi/bin/:$ENV{PATH}")
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12/")

#set(CUDA_TOOLKIT_ROOT_DIR "/home/linzhiheng/cuda-12.2/")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.2/")
set(MPI_HOME "/home/linzhiheng/thirdparty/openmpi/")
set(CUDA_HOME "/usr/local/cuda-12.2")
# set(CUDA_HOME "/home/linzhiheng/cuda-12.2/")
# set(CUDNN_HOME "/home/linzhiheng/thirdparty/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/")
#set(NVSHMEM_HOME "/home/linzhiheng/thirdparty/nvshmem_src_2.0.3-0/build/")
set(NVSHMEM_HOME "/home/linzhiheng/thirdparty/nvshmem_src_2.9.0-2/build/")
#set(NVSHMEM_HOME "/home/linzhiheng/thirdparty/nvshmem_src_2.10.1-3/build/")
set(NCCL_HOME "/home/linzhiheng/nccl/build/")

#############################################################################
set(CUDA_ARCHS 80)
set(SM_ARCH "sm_80")
set($ENV{NVCC_GENCODE} ${SM_ARCH})
set($ENV{NVSHMEM_USE_GDRCOPY} "0")

set(CMAKE_CUDA_ARCHITECTURES 80)
#############################################################################
#link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(${NVSHMEM_HOME}/lib)
# link_directories(${CUDNN_HOME}/lib64)
link_directories(${MPI_HOME}/lib)
link_directories(/usr/local/lib)
link_directories(${NCCL_HOME}/lib)
link_directories(${CUDA_HOME}/lib64)

add_definitions(-w)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Threads REQUIRED)

#############################################################################
add_executable(jupiter_gpm
    common/graph_partition.cc
    common/graph.cc
    common/scheduler.cc
    common/VertexSet.cc
    include/common.h
    include/graph.h
    include/graph_gpu.h
    include/graph_partition.h
    src/main.cc
    src/jupiter_gpm.cu
)

target_include_directories(jupiter_gpm 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    # ${CUDNN_HOME}/include
    include
    ${CUCO_HOME}/include
)

target_compile_options(jupiter_gpm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -fopenmp
                        # -std=c++11
                        -std=c++17
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                        --expt-relaxed-constexpr
                        --expt-extended-lambda
                       >)

set_target_properties(jupiter_gpm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(jupiter_gpm PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(jupiter_gpm
    nvshmem 
    cuda
    #mpi_cxx 
    mpi 
    cublas 
    # cudnn 
    gomp 
    curand

)
target_link_libraries(jupiter_gpm nvidia-ml)
