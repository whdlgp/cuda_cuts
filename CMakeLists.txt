cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project( cuda_cut LANGUAGES CXX CUDA)

# source directory
set( SRCS CudaCuts.cu CudaCuts_kernel.cu Example.cu )

# link
add_executable( cuda_cut.out ${SRCS} )

