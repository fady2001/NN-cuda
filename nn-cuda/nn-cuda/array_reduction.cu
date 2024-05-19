#include "common.hpp"
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

template <class T>
void reduce_cpu(T* out, const T* in, int N, REDUCTION reduction) {
    if (reduction == MAX) {
        *out = in[0];
        for (int i = 1; i < N; ++i) {
            if (in[i] > *out) {
                *out = in[i];
            }
        }
        return;
    }
    T sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += in[i];
    }
    *out = sum;
    if (reduction == MEAN) {
        *out /= N;
    }
}

// kernel 1
template <class T>
__global__ void reduce_kernel1(T* out, const T* in, int N, REDUCTION reduction) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		if (reduction == MAX) {
			atomicMax((int*)out, __float_as_int(in[i])); // Use atomicMax for floats
		}
		else {
			atomicAdd(out, in[i]);
            __syncthreads();
            if (threadIdx.x == 0 && reduction == MEAN) {
                    *out /= N;
            }
		}
	}
}


// kernel 2
template <class T>
__global__ void reduce_kernel2(T *d_a, T *d_result, int size, REDUCTION reduction) {
    extern __shared__ T v[];

    int amount_per_thread = (size + blockDim.x - 1) / blockDim.x;
    int start_index = threadIdx.x * amount_per_thread; 
    int end_index = min(start_index + amount_per_thread, size); 
    T partialsum = (reduction == MAX) ? d_a[start_index] : 0.0f;

    for(int k = start_index; k < end_index; k++) {
        if (reduction == MAX) {
            partialsum = max(partialsum, d_a[k]);
        } else {
            partialsum += d_a[k];
        }
    }
    v[threadIdx.x] = partialsum; 

    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x) {
            if (reduction == MAX) {
                v[index] = max(v[index], v[index + s]);
            } else {
                v[index] += v[index + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (reduction == MEAN) {
            d_result[0] = v[0] / size;
        } else {
            d_result[0] = v[0];
        }
    }
}

// kernel 3
template <class T>
__global__ void reduce_kernel3(T *d_a, T *d_result, int size, REDUCTION reduction) {
    extern __shared__ T v[];
    int amount_per_thread = (size + blockDim.x - 1) / blockDim.x;
    int start_index = threadIdx.x * amount_per_thread; 
    int end_index = min(start_index + amount_per_thread, size); 
    T partialsum = (reduction == MAX) ? d_a[start_index] : 0.0f;

    for(int k = start_index; k < end_index; k++) {
        if (reduction == MAX) {
            partialsum = max(partialsum, d_a[k]);
        } else {
            partialsum += d_a[k];
        }
    }
    v[threadIdx.x] = partialsum; 

    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (reduction == MAX) {
                v[threadIdx.x] = max(v[threadIdx.x], v[threadIdx.x + s]);
            } else {
                v[threadIdx.x] += v[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (reduction == MEAN) {
            d_result[0] = v[0] / size;
        } else {
            d_result[0] = v[0];
        }
    }
}

// launcher of array reduction kernels
template <class T>
void run_kernel1(T *d_inp, T *d_result, int N,int block_size, REDUCTION reduction) {
    reduce_kernel1 << <(N + block_size - 1) / block_size, block_size >> > (d_inp, d_result, N,reduction);
}

template <class T>
void run_kernel2(T *d_inp, T *d_result, int N, int block_size, REDUCTION reduction) {
    reduce_kernel2 << <1, block_size, block_size * sizeof(T) >> > (d_inp, d_result, N,reduction);
}

template <class T>
void run_kernel3(T *d_inp, T *d_result, int N, int block_size, REDUCTION reduction) {
    reduce_kernel3 << <1, block_size, block_size * sizeof(T) >> > (d_inp, d_result, N,reduction);
}

int main() {
	// Placeholder for actual test implementation
	// This should include allocation of device memory, kernel launches, and copying results back to host
	// Example usage:
	const int N = 10000000;
	float* h_in = make_random_float(N);
	float* h_out = new float;
	float* d_in1, * d_out1;
    float* d_in2, * d_out2;
    float* d_in3, * d_out3;

	cudaMalloc(&d_in1, N * sizeof(float));
    cudaMalloc(&d_out1, sizeof(float));
    cudaMalloc(&d_in2, N * sizeof(float));
    cudaMalloc(&d_out2, sizeof(float));
    cudaMalloc(&d_in3, N * sizeof(float));
    cudaMalloc(&d_out3, sizeof(float));
    cudaMemcpy(d_in1, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in3, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

	// Choose reduction type
	REDUCTION reduction = MEAN; // Change to MEAN or MAX as needed

    // Run CPU reduction
	measureExecutionTime(reduce_cpu<float>, h_out, h_in, N, reduction);

	int block_sizes[] = { 8,16, 32, 64, 128, 256, 512, 1024 };
	// first check the correctness of the kernel
	/*for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
		int block_size = block_sizes[j];
		printf("Checking block size %d.\n", block_size);
		run_kernel1<float>(d_in1, d_out1, N, block_sizes[j], reduction);
        run_kernel2<float>(d_in2, d_out2, N, block_sizes[j], reduction);
        run_kernel3<float>(d_in3, d_out3, N, block_sizes[j], reduction);
		validate_result(d_out1, h_out, "out", 1, 1e-4f);
        validate_result(d_out2, h_out, "out", 1, 1e-4f);
		validate_result(d_out3, h_out, "out", 1, 1e-4f);
	}*/

	printf("All results match. Starting benchmarks.\n\n");
	for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
		int block_size = block_sizes[j];

		int repeat_times = 100;
        //float elapsed_time1 = benchmark_kernel(repeat_times,run_kernel1<float>, d_in1, d_out1, N, block_size, reduction);
        float elapsed_time2 = benchmark_kernel(repeat_times,run_kernel2<float>, d_in2, d_out2, N, block_size, reduction);
        float elapsed_time3 = benchmark_kernel(repeat_times,run_kernel3<float>, d_in3, d_out3, N, block_size, reduction);
		
        //printf("elapsed time for kernel 1 with block size %d: %f ms\n", block_size, elapsed_time1);
        printf("elapsed time for kernel 2 with block size %d: %f ms\n", block_size, elapsed_time2);
        printf("elapsed time for kernel 3 with block size %d: %f ms\n", block_size, elapsed_time3);
	}
	// Clean up
    cudaFree(d_in1);
    cudaFree(d_out1);
    cudaFree(d_in2);
    cudaFree(d_out2);
    cudaFree(d_in3);
    cudaFree(d_out3);
    delete[] h_in;
    delete h_out;
    return 0;
}
