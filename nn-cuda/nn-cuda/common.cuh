#pragma once

#include "../vendor/npy.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// Function declarations
template <class T>
__host__ __device__ T ceil_div(T dividend, T divisor)
{
	return (dividend + divisor - 1) / divisor;
}

void cuda_check(cudaError_t error, const char *file, int line);
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

template <class T>
T* make_random_float_01(size_t N)
{
	T* arr = (T*)malloc(N * sizeof(T));
	for (size_t i = 0; i < N; i++)
	{
		arr[i] = ((T)rand() / RAND_MAX); // range 0..1
	}
	return arr;
}

template <class T>
T* make_random_float(size_t N)
{
	T* arr = (T*)malloc(N * sizeof(T));
	for (size_t i = 0; i < N; i++)
	{
		arr[i] = ((T)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
	}
	return arr;
}

int *make_random_int(size_t N, int V);

template<class T>
T* make_zeros_float(size_t N)
{
	T* arr = (T*)malloc(N * sizeof(T));
	memset(arr, 0, N * sizeof(T)); // all zero
	return arr;
}

template<class T>
T* make_ones_float(size_t N)
{
	T* arr = (T*)malloc(N * sizeof(T));
	for (size_t i = 0; i < N; i++)
	{
		arr[i] = 1.0f;
	}
	return arr;
}


struct NBArray
{
	float *data;
	int n_dims;
	int *shape;
};

template<class T>
void write_npy(const char* filename, const T* data, unsigned int n_dims, const unsigned long* shape)
{
	std::string full_path = "..\\with-torch-tests\\" + std::string(filename);
	npy::SaveArrayAsNumpy<T>(full_path, false, n_dims, shape, data);
}

void print_2D_Matrix(float *matrix, const char *name, int rows, int cols);

template <class T>
void validate_result(T *device_result, const T *cpu_reference, const char *name, std::size_t num_elements, T tolerance = 1e-4);

template <class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs &&...kernel_args);

void *malloc_check(size_t size, const char *file, int line);
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)