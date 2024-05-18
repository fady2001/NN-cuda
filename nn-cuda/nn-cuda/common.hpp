#pragma once

#include "../vendor/npy.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <chrono>
#include <functional>


#define uint unsigned int

enum REDUCTION { SUM, MEAN, MAX };

// Function declarations
template <class T> __host__ __device__ T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}
// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

float *make_random_float_01(size_t N) {
  float *arr = (float *)malloc(N * sizeof(float));
  for (size_t i = 0; i < N; i++) {
    arr[i] = ((float)rand() / RAND_MAX); // range 0..1
  }
  return arr;
}

float *make_random_float(size_t N) {
  float *arr = (float *)malloc(N * sizeof(float));
  for (size_t i = 0; i < N; i++) {
    arr[i] =
        ((float)rand() / RAND_MAX) * (float)2.0 - (float)1.0; // range -1..1
  }
  return arr;
}

uint *make_random_int(size_t N, int V) {
  uint *arr = (uint *)malloc(N * sizeof(uint));
  for (size_t i = 0; i < N; i++) {
    arr[i] = rand() % V; // range 0..V-1
  }
  return arr;
}

float *make_zeros_float(size_t N) {
  float *arr = (float *)malloc(N * sizeof(float));
  memset(arr, 0, N * sizeof(float)); // all zero
  return arr;
}

float *make_ones_float(size_t N) {
  float *arr = (float *)malloc(N * sizeof(float));
  for (size_t i = 0; i < N; i++) {
    arr[i] = 1.0f;
  }
  return arr;
}

template <class T>
void write_npy(const char *filename, const T *data, unsigned int n_dims,
               const unsigned long *shape) {
  std::string full_path = "..\\with-torch-tests\\" + std::string(filename);
  npy::SaveArrayAsNumpy<T>(full_path, false, n_dims, shape, data);
}

template <class dtype>
void read_npy(const char *filepath, dtype **data, uint **shape_v, uint *n_dims,
              bool pinnedMem = false) {
  std::string full_path = std::string(filepath);
  std::vector<dtype> vec = std::vector<dtype>();
  std::vector<unsigned long> shape_vec = std::vector<unsigned long>();
  npy::LoadArrayFromNumpy(full_path, shape_vec, vec);
  *n_dims = shape_vec.size();
  *shape_v = (uint *)malloc(*n_dims * sizeof(uint));
  for (uint i = 0; i < *n_dims; i++) {
    (*shape_v)[i] = shape_vec[i];
  }
  if (pinnedMem) {
    cudaCheck(cudaMallocHost(data, vec.size() * sizeof(dtype)));
  } else {
    *data = (dtype *)malloc(vec.size() * sizeof(dtype));
  }
  std::memcpy(*data, vec.data(), vec.size() * sizeof(dtype));
}

void get_from_gpu_and_print(const char *name, float *d_ptr, int size) {
  float *ptr = (float *)malloc(size * sizeof(float));
  cudaCheck(
      cudaMemcpy(ptr, d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));
  printf("%s\n", name);
  for (int i = 0; i < 10; i++) {
    printf("%f\n", ptr[i]);
  }
  printf("-----------------------\n");
  free(ptr);
}

void save_1d(const char *name, float *d_ptr, unsigned long size) {
  float *ptr = (float *)malloc(size * sizeof(float));
  cudaCheck(
      cudaMemcpy(ptr, d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));
  write_npy(name, ptr, 1, new unsigned long[1]{size});
  free(ptr);
}

void save_2d(const char *name, float *d_ptr, unsigned long size1,
             unsigned long size2) {
  float *ptr = (float *)malloc(size1 * size2 * sizeof(float));
  cudaCheck(cudaMemcpy(ptr, d_ptr, size1 * size2 * sizeof(float),
                       cudaMemcpyDeviceToHost));
  write_npy(name, ptr, 2, new unsigned long[2]{size1, size2});
  free(ptr);
}

void print_2D_Matrix(float *matrix, const char *name, int rows, int cols) {
  printf("Matrix: %s with size %d x %d: \n", name, rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
}

void validate_result(float *device_result, const float *cpu_reference,
                     const char *name, size_t num_elements, float tolerance) {
  float *out_gpu = (float *)malloc(num_elements * sizeof(float));
  cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(float),
                       cudaMemcpyDeviceToHost));

  int nfaults = 0;
  for (int i = 0; i < num_elements; i++) {
    if (i < 5) {
      printf("%f %f\n", cpu_reference[i], out_gpu[i]);
    }
    if (fabs(cpu_reference[i] - out_gpu[i]) > tolerance &&
        !isnan(cpu_reference[i])) {
      printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i,
             cpu_reference[i], out_gpu[i]);
      nfaults++;
      if (nfaults >= 10) {
        free(out_gpu);
        exit(EXIT_FAILURE);
      }
    }
  }
  free(out_gpu);
}

void *malloc_check(size_t size, const char *file, int line);
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

template <class Kernel, class... KernelArgs>
float run_kernel(Kernel kernel, KernelArgs &&...kernel_args) {
  cudaEvent_t start, stop;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
  cudaCheck(cudaEventRecord(start, nullptr));
  kernel(std::forward<KernelArgs>(kernel_args)...);
  cudaCheck(cudaEventRecord(stop, nullptr));
  cudaCheck(cudaEventSynchronize(stop));
  float elapsed_time;
  cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));
  return elapsed_time;
}

template <class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel,
                       KernelArgs &&...kernel_args) {
  cudaEvent_t start, stop;
  // prepare buffer to scrub L2 cache between benchmarks
  // just memset a large dummy array, recommended by
  // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
  // and apparently used in nvbench.
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp{};
  cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
  void *flush_buffer;
  cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
  float elapsed_time = 0.f;
  for (int i = 0; i < repeats; i++) {
    // clear L2
    cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
    // now we can start recording the timing of the kernel
    cudaCheck(cudaEventRecord(start, nullptr));
    kernel(std::forward<KernelArgs>(kernel_args)...);
    cudaCheck(cudaEventRecord(stop, nullptr));
    cudaCheck(cudaEventSynchronize(start));
    cudaCheck(cudaEventSynchronize(stop));
    float single_call;
    cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
    elapsed_time += single_call;
  }

  cudaCheck(cudaFree(flush_buffer));

  return elapsed_time / repeats;
}
struct trainData {
  float *inp;
  uint *input_shape; // 2D array (#datapoints , #features)
  uint *target;
  uint *target_shape; // 1D array (#datapoints)
};

void read_training_data(const char *inp_path, const char *target_path,
                        trainData &data, bool verbose = false,
                        bool pinnedMem = false) {
  uint n_dims;
  read_npy<float>(inp_path, &data.inp, &data.input_shape, &n_dims, pinnedMem);
  if (n_dims != 2) {
    printf("Input data should be 2D\n");
    exit(1);
  }
  read_npy<uint>(target_path, &data.target, &data.target_shape, &n_dims,
                 pinnedMem);
  if (n_dims != 1) {
    printf("Target data should be 1D\n");
    exit(1);
  }
  if (verbose) {
    printf("Input data shape: %d x %d\n", data.input_shape[0],
           data.input_shape[1]);
    printf("Target data shape: %d\n", data.target_shape[0]);
    printf("First 10 elements of input data:\n");
    for (int i = 0; i < 10; i++) {
      printf("%f ", data.inp[i]);
    }
    printf("\nFirst 10 elements of target data:\n");
    for (int i = 0; i < 10; i++) {
      printf("%d ", data.target[i]);
    }
  }
}