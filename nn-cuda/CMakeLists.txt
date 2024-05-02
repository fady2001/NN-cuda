cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(nn-cuda LANGUAGES CXX CUDA)

# Set C++14 as standard for the host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDA 12.3 REQUIRED)
# Add the path to the include directory and other necessary directories
include_directories(include vendor)

# Collect all .cu files in the nn-cuda directory
file(GLOB CUDA_FILES nn-cuda/*.cu)

# Collect all .cpp files in the nn-cuda directory
file(GLOB CPP_FILES nn-cuda/*.cpp)

# Combine CUDA and C++ files into a single list
set(SOURCES ${CPP_FILES} "nn-cuda/common.cuh")

# Create the executable from all source files
add_executable(nn-cuda "./nn-cuda/main.cu" ${SOURCES})

# add_executable(lin "./nn-cuda/linear_forward.cu" ${SOURCES})

# If you have CUDA libraries to link
 target_link_libraries(nn-cuda ${CUDA_LIBRARIES})
