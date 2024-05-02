#pragma once

#include "../vendor/npy.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// Function declarations
template <class T>
__host__ __device__ T ceil_div(T dividend, T divisor);

void cuda_check(cudaError_t error, const char *file, int line);
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

float *make_random_float_01(size_t N);
float *make_random_float(size_t N);
int *make_random_int(size_t N, int V);
float *make_zeros_float(size_t N);
float *make_ones_float(size_t N);

struct NBArray
{
	float *data;
	int n_dims;
	int *shape;
};
void write_npy(const char *filename, const float *data, unsigned int n_dims, const unsigned long *shape);
void write_npy(const char *filename, const int *data, unsigned int n_dims, const unsigned long *shape);
void print_2D_Matrix(float *matrix, const char *name, int rows, int cols);

template <class T>
void validate_result(T *device_result, const T *cpu_reference, const char *name, std::size_t num_elements, T tolerance = 1e-4);

template <class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs &&...kernel_args);

void *malloc_check(size_t size, const char *file, int line);
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)