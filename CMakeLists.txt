cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project( cuda_cut LANGUAGES CXX CUDA)

# source directory
set( SRCS CudaCuts.cu CudaCuts_kernel.cu Example.cu )

# OpenCV 
set( ld_flag /home/choyg/Desktop/workspace/opencv_ffmpeg/opencv-3.4.10/install/lib )
set( include_flag /home/choyg/Desktop/workspace/opencv_ffmpeg/opencv-3.4.10/install/include )
set( libs opencv_img_hash opencv_world )

include_directories( ${include_flag} )
link_directories( ${ld_flag} )

# link
add_executable( cuda_cut.out ${SRCS} )

target_link_libraries( cuda_cut.out ${libs} )

