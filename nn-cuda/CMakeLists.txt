cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Set the runtime output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set CUDA architectures if not defined
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif ()

project(nn-cuda LANGUAGES CXX CUDA)

# set flags for nvcc
# set(CMAKE_CUDA_FLAGS "-Wall")

# Set C++14 as standard for the host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the path to the include directory and other necessary directories
include_directories(include vendor)

# Collect all .cu files in the nn-cuda directory
# file(GLOB CUDA_FILES nn-cuda/*.cu)

# Collect all .cpp files in the nn-cuda directory
file(GLOB CPP_FILES nn-cuda/*.cpp)

#
# Combine CUDA and C++ files into a single list
set(SOURCES ${CPP_FILES})

# Create the executable from all source files
# add_executable(nn-cuda "./nn-cuda/main.cu" ${SOURCES})
add_executable(nn-cuda "nn-cuda/main_training.cu" ${SOURCES}
        nn-cuda/main_training.cu)

# Set include directories for the target
target_include_directories(nn-cuda PRIVATE include vendor)

# Set CUDA compilation flags (uncomment if needed)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall")

# Link CUDA libraries if needed (uncomment if needed)
# target_link_libraries(nn-cuda ${CUDA_LIBRARIES})
add_executable(training nn-cuda/main_training.cu)
