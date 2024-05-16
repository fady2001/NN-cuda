#pragma once

#include "../vendor/npy.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

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

int *make_random_int(size_t N, int V) {
  int *arr = (int *)malloc(N * sizeof(int));
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

template<class T>
void write_npy(const char *filename, const T *data, unsigned int n_dims,
               const unsigned long *shape) {
  std::string full_path = "..\\with-torch-tests\\" + std::string(filename);
  npy::SaveArrayAsNumpy<T>(full_path, false, n_dims, shape, data);
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